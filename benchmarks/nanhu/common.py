"""Portable file, symbol binding, and accepted-observation checks."""
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import struct


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def relative_file(root, name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and path.parts,
            "Unsafe manifest path: " + name)
    root = Path(root).resolve()
    target = root.joinpath(*path.parts).resolve()
    require(target.is_relative_to(root), "Manifest path escapes root: " + name)
    return target


def verify_files(root, files):
    for name, expected in files.items():
        path = relative_file(root, name)
        require(path.is_file(), "Missing file (run git lfs pull): " + name)
        require(identity(path) == expected, "File identity differs: " + name)
    return len(files)


def symbol_map(path, wanted=None):
    """Read the fixed little-endian symbol ABI without retaining huge maps."""
    result = {}
    with Path(path).open("rb") as stream:
        def take(n):
            data = stream.read(n)
            require(len(data) == n, "Truncated symbol map")
            return data

        def u64():
            return struct.unpack("<Q", take(8))[0]

        # The file contains register pairs, then signal and memory triples.
        for map_index, element_bytes in enumerate((8, 12, 12)):
            seen = set()
            count = u64()
            require(count < 10_000_000, "Invalid symbol count")
            for _ in range(count):
                size = u64()
                require(0 < size < 1_000_000, "Invalid symbol name size")
                name = take(size).decode("utf-8")
                require(name not in seen, "Duplicate symbol: " + name)
                seen.add(name)
                fragments = u64()
                require(fragments < 1_000_000, "Invalid fragment count")
                data = take(fragments * element_bytes)
                if map_index == 0 and (wanted is None or name in wanted):
                    result[name] = list(struct.iter_unpack("<II", data))
        require(not stream.read(1), "Trailing bytes in symbol map")
    if wanted is not None:
        require(set(result) == set(wanted), "Missing watch symbols: " + str(set(wanted) - set(result)))
    return result


def validate_fragments(name, fragments, width, pool_bytes):
    require(0 < len(fragments) <= 16 and sum(b for _, b in fragments) == width
            and all(0 <= a < pool_bytes and 0 < b <= 4 for a, b in fragments),
            "Invalid symbol fragments: " + name)


def required_io():
    core = "SimTop.cpu.l_soc.core_with_l2.core."
    ram = "SimTop.cpu.l_simAXIMem.ram.rdata_mem.helper_"
    flash, sd = "SimTop.cpu.l_simMMIO.flash.flash.", "SimTop.cpu.l_simMMIO.sd.sdHelper."
    trap = core + "ctrlBlock_rob_difftest_module_6.dpic."
    event = core + "exuBlocks_fuBlock.exeUnits_6.csr.difftest_module.dpic."
    specs = {"SimTop.reset": 1, "SimTop.difftest_uart_in_ch": 8,
             "SimTop.difftest_uart_out_valid": 1, "SimTop.difftest_uart_out_ch": 8,
             flash + "ren": 1, flash + "addr": 16, flash + "data": 64,
             sd + "ren": 1, sd + "setAddr": 1, sd + "data": 32,
             trap + "io_hasTrap": 1, trap + "io_instrCnt": 64, trap + "io_cycleCnt": 64,
             event + "io_exception": 32, event + "io_interrupt": 32,
             ram + "0.r_0_enable": 1, ram + "0.w_0_enable": 1}
    specs.update({core + "exuBlocks_scheduler_difftest_module.dpic.io_value_" + str(i): 64
                  for i in range(32)})
    for lane in range(6):
        prefix = core + "ctrlBlock_rob_difftest_module" + (f"_{lane}" if lane else "") + ".dpic."
        specs.update({prefix + key: 1 for key in ("enable", "io_skip", "io_isRVC", "io_rfwen", "io_fpwen")})
        specs.update({prefix + key: 8 for key in ("io_index", "io_special", "io_wdest")})
    for port in range(4):
        specs.update({ram + str(port) + "." + key: 64 for key in
                      ("r_0_data", "r_0_index", "w_0_data", "w_0_index", "w_0_mask")})
    return specs


def bind_watches(model_dir, watches, pool_bytes):
    names = {item["name"] for item in watches}
    require(len(names) == len(watches) == 6, "Expected six unique watches")
    full = symbol_map(Path(model_dir) / "GPUSimSymbols.bin", names)
    io = symbol_map(Path(model_dir) / "GPUSimIOSymbols.bin")
    for name, width in required_io().items():
        require(name in io, "Missing required IO: " + name)
        validate_fragments(name, io[name], width, pool_bytes)
    rows = []
    for item in watches:
        name, width = item["name"], item["width"]
        fragments = full[name]
        validate_fragments(name, fragments, width, pool_bytes)
        require(name not in io or io[name] == fragments, "Conflicting IO watch: " + name)
        rows.append(f"{name} {width} {len(fragments)} " +
                    " ".join(f"{a} {b}" for a, b in fragments))
    return "\n".join(rows) + "\n"


ENDPOINT_KEYS = ("total_cycles", "reset_cycles", "first_commit_cycle", "trap_cycle",
                 "commit_records", "retired_with_fusion", "loop_count",
                 "register_comparisons", "ram_reads", "ram_writes", "flash_reads",
                 "observed_commit_pc_instruction", "registers")


def first_difference(actual, expected, binary=False):
    offset = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                  min(len(actual), len(expected)))
    if binary:
        return f"byte {offset}, watch record {offset // 56}, record byte {offset % 56}"
    return f"line {actual[:offset].count(bytes([10])) + 1}, byte {offset}"


def validate_result(output, loops, mode, reference_root, expected):
    output, reference_root = Path(output), Path(reference_root)
    value = read_json(output / "result.json")
    require(value.get("status") == "passed", "Runner status is not passed")
    require(len(value.get("registers", [])) == 32, "Expected all 32 GPRs")
    if mode == "checked":
        endpoint = {key: expected[key] for key in ENDPOINT_KEYS}
    else:
        endpoint = {key: expected[key] for key in
                    ("total_cycles", "first_commit_cycle", "trap_cycle", "loop_count",
                     "ram_reads", "ram_writes", "flash_reads", "registers")}
        endpoint.update(mode="batch-final", retired=expected["retired_with_fusion"],
                        records=expected["commit_records"], observation_downloads=1,
                        cpu_cycle_checks=0, launches=1)
    for key, want in endpoint.items():
        require(value.get(key) == want,
                f"Endpoint {key}: got {value.get(key)!r}, expected {want!r}")
    for name in ("commits.tsv", "devices.tsv", "watch.bin"):
        actual = (output / name).read_bytes()
        if mode == "checked":
            with gzip.open(reference_root / f"loop{loops}" / (name + ".gz"), "rb") as stream:
                accepted = stream.read()
            require(actual == accepted, name + " differs at " +
                    first_difference(actual, accepted, name == "watch.bin"))
            if name == "watch.bin":
                require(len(actual) == 169 * 56, "Expected 169 watch records")
        else:
            require(not actual, "Batch timing unexpectedly collected " + name)
    evaluation, wall = value["evaluation_seconds"], value["wall_seconds"]
    require(math.isfinite(evaluation) and math.isfinite(wall) and 0 < evaluation <= wall,
            "Invalid evaluation/wall timing")
    if mode != "checked":
        require(math.isclose(value["cycles_per_second"], expected["total_cycles"] / wall,
                             rel_tol=1e-7, abs_tol=1e-5), "Cycle-rate denominator differs")
    return value


def paired_summary(rows, baseline, candidate):
    pairs = []
    indexes = sorted({row["pair"] for row in rows})
    require(len(indexes) >= 3, "At least three pairs required")
    for index in indexes:
        sample = [row for row in rows if row["pair"] == index]
        require(len(sample) == 2 and {r["variant"] for r in sample} == {baseline, candidate},
                f"Incomplete or duplicate pair {index}")
        by_variant = {row["variant"]: row for row in sample}
        a, b = by_variant[baseline]["result"], by_variant[candidate]["result"]
        for result in (a, b):
            require(all(math.isfinite(result[k]) and result[k] > 0
                        for k in ("evaluation_seconds", "wall_seconds")), "Invalid paired timing")
        pairs.append({"pair": index, "order": [row["variant"] for row in sample],
                      "evaluation_speedup": a["evaluation_seconds"] / b["evaluation_seconds"],
                      "wall_speedup": a["wall_seconds"] / b["wall_seconds"]})
    return {"pairs": pairs, "median_paired_evaluation_speedup": statistics.median(
        pair["evaluation_speedup"] for pair in pairs),
        "median_paired_wall_speedup": statistics.median(pair["wall_speedup"] for pair in pairs)}

#!/usr/bin/env python3
"""Import small accepted evidence and freeze portable model identities, not models."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from common import ENDPOINT_KEYS, identity, require, write_json

SWEEP_SHA = "c8a972f2af8bdfaf2950d843a25b8f5473d99b049223cefd6f8b6cd51ab0f206"
LEGACY_SHA = "e36e13a4492f16f7f863e076de2d39bdd427f6ec14f56d9e5cc531a2995d3623"
COMPILER_SHA = "3f524ae3aa571648f0433fd998e95e94d494e020abc2dd9aa10b7eda4fc07398"
INPUT_SHA = "87424bb2a3406cf8dbf256af82b2a305a70d3e91f7c6ad37b04f24237ffd2f04"


class Evidence:
    def __init__(self, path, sha):
        require(identity(path)["sha256"] == sha, "Unexpected evidence archive")
        self.tar = tarfile.open(path, "r:gz")
        self.manifest = json.load(self.tar.extractfile("MANIFEST.json"))["files"]

    def read(self, name):
        data = self.tar.extractfile(name).read()
        require({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} ==
                self.manifest[name], "Evidence member differs: " + name)
        return data

    def json(self, name):
        return json.loads(self.read(name))


def import_data(output, sweep, legacy, models):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files, references, endpoints = {}, {}, {}

    def save(name, data):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            require(path.read_bytes() == data, "Refusing to replace differing " + name)
        else:
            path.write_bytes(data)
        files[name] = identity(path)

    for loops in (1, 1000):
        source = f"accepted-p20/loop{loops}/"
        result = sweep.json(source + "result.json")
        endpoints[str(loops)] = {key: result[key] for key in ENDPOINT_KEYS}
        for name in ("commits.tsv", "devices.tsv", "watch.bin"):
            data = sweep.read(source + name)
            require(data == sweep.read(f"accepted-v5/loop{loops}/{name}"),
                    "Accepted v5/p20 reference disagreement")
            if name == "commits.tsv":
                require(data == sweep.read(f"RTL-reference/verilator-bump/loop{loops}.trace"),
                        "Original RTL commit reference differs")
            dest = f"references/loop{loops}/{name}.gz"
            save(dest, gzip.compress(data, compresslevel=9, mtime=0))
            references[dest] = {"uncompressed_bytes": len(data),
                                "uncompressed_sha256": hashlib.sha256(data).hexdigest()}
    for name in ("checked-loop-1.bin", "checked-loop-1000.bin", "checked-loop.S", "checked-loop.ld"):
        save("workloads/" + name, sweep.read("experiment/workloads/p20/" + name))
    watches = [{"name": row.split()[0], "width": int(row.split()[1])}
               for row in sweep.read("experiment/workloads/p20/nanhu-watch.tsv").decode().splitlines()]
    write_json(output / "watches.json", watches)
    write_json(output / "endpoints.json", endpoints)
    for name in ("watches.json", "endpoints.json"):
        files[name] = identity(output / name)
    write_json(output / "reference_manifest.json", {
        "schema_version": 1, "files": files, "references": references,
        "evidence_archive_sha256": SWEEP_SHA, "watch_records": 169,
        "watch_record_bytes": 56, "pc_instruction_columns": "oracle_not_observed_DUT",
        "cross_layout_pool_comparison": False})
    old = legacy.json("experiment/input-freeze.json")
    frozen = sweep.json("experiment/input-freeze.json")
    structures = sweep.json("experiment/structure.json")
    layouts = {}
    for variant in ("p10", "p20", "p25", "p30", "p40"):
        narrow = variant == "p10"
        freeze = old if narrow else frozen
        key = "narrow16" if narrow else variant
        source = models[variant]
        payloads = {}
        for name in ("GPUSimDesign.bin", "GPUSimSymbols.bin", "GPUSimIOSymbols.bin"):
            remote = freeze["designs"][key] + "/" + name
            expected = freeze["files"][remote]
            local = source / name
            if name == "GPUSimDesign.bin" and not local.exists():
                local = source / "GPUSimDesign.wide32.bin"
            require(identity(local) == expected, "Model differs from accepted freeze: " + str(local))
            payloads[f"models/{variant}/{name}"] = expected
        st = ({"rounds": [944, 8583], "max_local_bytes": 59983,
               "regpool_bytes": 1064832} if narrow else structures[variant])
        layouts[variant] = {
            "part_size": int(variant[1:]) / 100, "abi": "legacy16" if narrow else "wide32",
            "directory": f"models/{variant}", "files": payloads,
            "pool_budget_bytes": 65534 if narrow else 98304,
            "round_partitions": st["rounds"], "max_local_bytes": st["max_local_bytes"],
            "regpool_bytes": st["regpool_bytes"]}
    write_json(output / "model_manifest.json", {
        "schema_version": 1, "baseline": "p20", "cut_target": 0.70,
        "native_repartitions": [0, 0], "capacity_repair": False,
        "source_flattened_sha256": INPUT_SHA, "wide_compiler_sha256": COMPILER_SHA,
        "acceptance_evidence_sha256": {"legacy_and_p20": LEGACY_SHA, "larger_sweep": SWEEP_SHA},
        "layouts": layouts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-evidence", required=True, type=Path)
    parser.add_argument("--legacy-evidence", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, metavar="VARIANT=DESIGN_DIR")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    models = dict(item.split("=", 1) for item in args.model)
    require(set(models) == {"p10", "p20", "p25", "p30", "p40"}, "Specify all five model directories")
    import_data(args.output, Evidence(args.sweep_evidence, SWEEP_SHA),
                Evidence(args.legacy_evidence, LEGACY_SHA), {k: Path(v) for k, v in models.items()})
    print("Imported accepted references and froze all 15 model file identities; no models copied.")


if __name__ == "__main__":
    main()

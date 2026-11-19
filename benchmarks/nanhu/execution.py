"""Recorded action execution; GPU reservation belongs to the external scheduler."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from common import (identity, read_json, relative_file, require, validate_result,
                    verify_files, write_json)


def environment():
    query = ["nvidia-smi", "--query-gpu=uuid,name,driver_version,compute_cap,memory.total",
             "--format=csv,noheader"]
    try:
        probe = subprocess.run(query, capture_output=True, text=True, timeout=30)
        gpu = {"returncode": probe.returncode, "stdout": probe.stdout, "stderr": probe.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu = {"error": str(error)}
    return {"platform": platform.platform(), "python": platform.python_version(),
            "CPU_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "GPU_query": gpu}


class Runtime:
    def __init__(self, build_root, abi):
        self.root = Path(build_root).resolve() / abi
        self.manifest_path = self.root / "build.json"
        self.manifest = read_json(self.manifest_path)
        require(self.manifest["abi"] == abi and self.manifest.get("status") == "built",
                "Runtime ABI/build status differs: " + abi)
        self.fingerprint = identity(self.manifest_path)

    def binary(self, mode):
        name = self.manifest["binaries"][mode]
        path = relative_file(self.root, name)
        key = {"checked": "checked_binary", "batch_final": "final_binary",
               "resources": "resource_probe"}[mode]
        expected = {k: self.manifest[key][k] for k in ("bytes", "sha256")}
        require(identity(path) == expected, "Runtime binary changed: " + str(path))
        return path


def command(package, prepared, runtime, variant, loops, output, checked):
    mode = "checked" if checked else "batch_final"
    limit = (20000 if loops == 1 else 100000) if checked else package.endpoints[str(loops)]["total_cycles"]
    app = [str(runtime.binary(mode)), str(package.design(variant)),
           str(Path(prepared).resolve() / variant / f"checked-loop-{loops}.bin"),
           str(Path(output).resolve() / "output"), str(limit), str(loops)]
    if not checked:
        app.append("batch-final")
    return app


def run_action(package, runtime, variant, loops, kind, app, directory, env, timeout):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    mode = "checked" if kind == "checked" else "batch_final"
    binary = runtime.binary("resources" if kind == "resources" else mode)
    binary_id = identity(binary)
    receipt = {"status": "running", "kind": kind, "variant": variant, "loops": loops,
               "command": app, "contract": package.contract(variant),
               "build_manifest": runtime.fingerprint, "binary": binary_id,
               "environment": env, "utc": datetime.now(timezone.utc).isoformat(),
               "cross_layout_pool_comparison": False, "pc_instruction_columns": "oracle_not_observed_DUT"}
    write_json(directory / "action.json", receipt)
    started = time.monotonic()
    try:
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            process = subprocess.run(app, stdout=stdout, stderr=stderr, timeout=timeout)
        receipt["process_seconds"] = time.monotonic() - started
        require(process.returncode == 0,
                f"Action exit {process.returncode}; inspect {directory / 'stderr.log'}")
        if kind == "resources":
            lines = (directory / "stdout.log").read_text().splitlines()
            data = next(line.split("=", 1)[1] for line in lines if line.startswith("RESOURCE_JSON="))
            result = json.loads(data)
            layout = package.layouts[variant]
            require(result["dynamic_shared_bytes"] == layout["max_local_bytes"] and
                    result["max_round_partitions"] == max(layout["round_partitions"]),
                    "Actual resource allocation differs from model")
            require(result["cooperative_grid"] > 0 and all(
                kernel["resident_blocks_per_SM"] > 0 for kernel in result["kernels"]),
                "No cooperative residency at actual shared size")
            write_json(directory / "resources.json", result)
            files = {"resources.json": identity(directory / "resources.json")}
        else:
            result = validate_result(directory / "output", loops, mode, package.root / "references",
                                     package.endpoints[str(loops)])
            files = {"output/" + name: identity(directory / "output" / name) for name in
                     ("result.json", "commits.tsv", "devices.tsv", "watch.bin")}
        require(identity(binary) == binary_id, "Binary changed during action")
        require(identity(runtime.manifest_path) == runtime.fingerprint, "Build changed during action")
        receipt.update(status="passed", result=result, files=files)
    except BaseException as error:
        receipt.update(status="failed", error=str(error), process_seconds=time.monotonic() - started)
        raise
    finally:
        write_json(directory / "action.json", receipt)
    return receipt


def checked_gate(package, checked_root, runtime, variant, env):
    require(env["GPU_query"].get("returncode") == 0, "Cannot verify current GPU identity")
    for loops in (1, 1000):
        directory = Path(checked_root) / f"checked-{variant}-loop{loops}"
        receipt = read_json(directory / "action.json")
        require(receipt["status"] == "passed" and receipt["kind"] == "checked" and
                receipt["variant"] == variant and receipt["loops"] == loops,
                "Missing accepted checked action: " + str(directory))
        require(receipt["contract"] == package.contract(variant) and
                receipt["build_manifest"] == runtime.fingerprint and
                receipt["binary"] == identity(runtime.binary("checked")),
                "Checked input/build changed: " + variant)
        for key in ("GPU_query", "CUDA_VISIBLE_DEVICES", "CPU_affinity"):
            require(receipt["environment"][key] == env[key], "Checked environment changed: " + key)
        verify_files(directory, receipt["files"])
        validate_result(directory / "output", loops, "checked", package.root / "references",
                        package.endpoints[str(loops)])

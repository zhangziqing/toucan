"""CPU-only full action flow using tiny synthetic models and fake executables."""
import argparse
import contextlib
import io
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch

from common import identity, read_json, required_io, write_json
from execution import Runtime, checked_gate
from nanhu import plan_bench, plan_checks, run_plan
from package import Package, assemble


FAKE_RUNNER = '''#!/usr/bin/env python3
import gzip,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
variant=Path(sys.argv[1]).name
if len(sys.argv)==2:
 layout=json.loads((root/'model_manifest.json').read_text())['layouts'][variant]
 print('RESOURCE_JSON='+json.dumps({'dynamic_shared_bytes':layout['max_local_bytes'],
  'max_round_partitions':max(layout['round_partitions']),'cooperative_grid':1,
  'kernels':[{'resident_blocks_per_SM':1}]}))
 sys.exit(0)
loops=int(sys.argv[5]);out=Path(sys.argv[3]);out.mkdir(parents=True)
value=json.loads((root/'endpoints.json').read_text())[str(loops)]
checked=len(sys.argv)==6
value.update(status='passed',evaluation_seconds=1.0,wall_seconds=2.0,
 cycles_per_second=value['total_cycles']/2)
if not checked:
 value.update(mode='batch-final',retired=value['retired_with_fusion'],records=value['commit_records'],
  observation_downloads=1,cpu_cycle_checks=0,launches=1)
(out/'result.json').write_text(json.dumps(value))
for name in ['commits.tsv','devices.tsv','watch.bin']:
 data=gzip.decompress((root/'references'/f'loop{loops}'/(name+'.gz')).read_bytes()) if checked else b''
 (out/name).write_bytes(data)
'''


def symbols(specs):
    data = bytearray(struct.pack("<Q", len(specs)))
    for name, width in specs.items():
        encoded = name.encode()
        count = (width + 3) // 4
        data.extend(struct.pack("<Q", len(encoded)) + encoded + struct.pack("<Q", count))
        for index in range(count):
            data.extend(struct.pack("<II", index, min(4, width - index * 4)))
    return data + bytes(16)


class Flow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = Path(__file__).resolve().parent
        for name in ("references", "workloads"):
            shutil.copytree(source / name, self.root / name)
        for name in ("reference_manifest.json", "endpoints.json", "watches.json"):
            shutil.copyfile(source / name, self.root / name)
        manifest = read_json(source / "model_manifest.json")
        manifest["layouts"] = {k: manifest["layouts"][k] for k in ("p20", "p40")}
        watches = {item["name"]: item["width"] for item in read_json(source / "watches.json")}
        for variant, layout in manifest["layouts"].items():
            directory = self.root / "models" / variant
            directory.mkdir(parents=True)
            (directory / "GPUSimDesign.bin").write_bytes(b"TCWIDE1\0synthetic")
            (directory / "GPUSimSymbols.bin").write_bytes(symbols(watches))
            (directory / "GPUSimIOSymbols.bin").write_bytes(symbols(required_io()))
            layout["files"] = {f"models/{variant}/{path.name}": identity(path)
                               for path in directory.iterdir()}
        write_json(self.root / "model_manifest.json", manifest)
        binaries = {"checked": "checked/nanhu-checked", "batch_final": "final/nanhu-final",
                    "resources": "final/resource-probe"}
        build = self.root / "build" / "wide32"
        metadata = {"status": "built", "abi": "wide32", "arch": "sm_86", "binaries": binaries}
        for mode, name in binaries.items():
            path = build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(FAKE_RUNNER)
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
            metadata[{"checked": "checked_binary", "batch_final": "final_binary",
                      "resources": "resource_probe"}[mode]] = identity(path)
        write_json(build / "build.json", metadata)
        self.package = Package(self.root)
        self.package.prepare(["p20", "p40"], self.root / "prepared")
        self.env = {"GPU_query": {"returncode": 0, "stdout": "synthetic CPU fixture"},
                    "CUDA_VISIBLE_DEVICES": "fixture", "CPU_affinity": [0]}

    def args(self, kind):
        return argparse.Namespace(command=kind, prepared_root=self.root / "prepared",
                                  build_root=self.root / "build", output=self.root / kind,
                                  timeout=10, dry_run=False, checked_root=self.root / "check",
                                  baseline="p20")

    def test_checked_then_paired_bench_and_gates(self):
        with patch("nanhu.environment", return_value=self.env), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_plan(self.package, self.args("check"), plan_checks(["p20", "p40"])), 0)
            self.assertEqual(run_plan(self.package, self.args("bench"), plan_bench(["p40"], "p20", 3)), 0)
        summary = read_json(self.root / "bench/summary.json")
        self.assertEqual(summary["warmups_excluded"], 2)
        self.assertEqual(summary["candidates"]["p40"]["median_paired_evaluation_speedup"], 1)
        runtime = Runtime(self.root / "build", "wide32")
        checked_gate(self.package, self.root / "check", runtime, "p20", self.env)
        (self.root / "check/checked-p20-loop1/output/watch.bin").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            checked_gate(self.package, self.root / "check", runtime, "p20", self.env)

    def test_profile_plan_and_binary_tamper(self):
        args = self.args("profile")
        args.dry_run, args.tool, args.profiler, args.profile_arg = True, "ncu", None, []
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            run_plan(self.package, args, [{"label": "profile-p40", "kind": "profile",
                                          "variant": "p40", "loops": 1000}])
        self.assertIn('"application"', capture.getvalue())
        self.assertFalse(args.output.exists())
        (self.root / "build/wide32/final/nanhu-final").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "binary changed"):
            Runtime(self.root / "build", "wide32").binary("batch_final")

    def test_movable_package_assembly(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / "source"
            runtime.mkdir()
            (runtime / "build_nanhu.py").write_text("# source fixture\n")
            output = Path(temp) / "assembled"
            assemble(self.package, runtime, output)
            self.assertEqual(Package(output).verify(["p20", "p40"]), 18)
            self.assertTrue((output / "runtime/build_nanhu.py").is_file())


if __name__ == "__main__":
    unittest.main()

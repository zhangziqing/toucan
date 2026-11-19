"""CPU-only negative controls for the portable Nanhu contract."""
import gzip
from pathlib import Path
import tempfile
import struct
import unittest

from common import (ENDPOINT_KEYS, identity, paired_summary, relative_file,
                    symbol_map, validate_result, verify_files, write_json)


class Contracts(unittest.TestCase):
    def test_symbol_abi_sections_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "symbols"
            def entry(name, values):
                return (struct.pack("<Q", 1) + struct.pack("<Q", len(name)) + name +
                        struct.pack("<Q", 1) + struct.pack("<" + "I" * len(values), *values))
            data = entry(b"reg", (3, 4)) + entry(b"signal", (1, 2, 3)) + entry(b"memory", (4, 5, 6))
            path.write_bytes(data)
            self.assertEqual(symbol_map(path, {"reg"}), {"reg": [(3, 4)]})
            with self.assertRaisesRegex(ValueError, "Missing watch"):
                symbol_map(path, {"absent"})
            path.write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError, "Truncated"):
                symbol_map(path)

    def test_manifest_tamper_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "payload").write_bytes(b"accepted")
            files = {"payload": identity(root / "payload")}
            self.assertEqual(verify_files(root, files), 1)
            (root / "payload").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                verify_files(root, files)
            for name in ("../outside", "/absolute"):
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    relative_file(root, name)
            (root / "link").symlink_to("/etc/passwd")
            with self.assertRaisesRegex(ValueError, "escapes"):
                relative_file(root, "link")

    def test_complete_checked_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, reference = root / "actual", root / "references"
            output.mkdir()
            (reference / "loop1").mkdir(parents=True)
            expected = dict.fromkeys(ENDPOINT_KEYS, 0)
            expected.update(registers=[0] * 32, loop_count=1, total_cycles=8511,
                            observed_commit_pc_instruction=False)
            result = dict(expected, status="passed", evaluation_seconds=1, wall_seconds=2)
            write_json(output / "result.json", result)
            for name, data in (("commits.tsv", b"commit\n"), ("devices.tsv", b"device\n"),
                               ("watch.bin", bytes(169 * 56))):
                (output / name).write_bytes(data)
                with gzip.open(reference / "loop1" / (name + ".gz"), "wb") as stream:
                    stream.write(data)
            validate_result(output, 1, "checked", reference, expected)
            (output / "commits.tsv").write_bytes(b"wrong\n")
            with self.assertRaisesRegex(ValueError, "commits.tsv differs"):
                validate_result(output, 1, "checked", reference, expected)
            result["registers"][7] = 1
            write_json(output / "result.json", result)
            with self.assertRaisesRegex(ValueError, "Endpoint registers"):
                validate_result(output, 1, "checked", reference, dict(expected, registers=[0] * 32))

    def test_pair_recomputation(self):
        rows = []
        for index, ratio in enumerate((2.0, 4.0, 3.0), 1):
            order = ("p20", "p40") if index % 2 else ("p40", "p20")
            for variant in order:
                seconds = ratio if variant == "p20" else 1
                rows.append({"pair": index, "variant": variant,
                             "result": {"evaluation_seconds": seconds, "wall_seconds": seconds}})
        result = paired_summary(rows, "p20", "p40")
        self.assertEqual(result["median_paired_evaluation_speedup"], 3)
        self.assertEqual(result["pairs"][1]["order"], ["p40", "p20"])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            paired_summary(rows[:-1], "p20", "p40")


if __name__ == "__main__":
    unittest.main()

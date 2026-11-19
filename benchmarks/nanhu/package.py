"""Portable model selection, preparation, and package assembly."""
from pathlib import Path
import shutil

from common import (bind_watches, identity, read_json, relative_file, require,
                    verify_files, write_json)


class Package:
    def __init__(self, root, artifact_root=None):
        self.root = Path(root).resolve()
        self.artifacts = Path(artifact_root or root).resolve()
        self.manifest = read_json(self.root / "model_manifest.json")
        self.references = read_json(self.root / "reference_manifest.json")
        require(self.manifest["schema_version"] == self.references["schema_version"] == 1,
                "Unsupported manifest schema")
        self.layouts = self.manifest["layouts"]
        self.endpoints = read_json(self.root / "endpoints.json")
        self.watches = read_json(self.root / "watches.json")

    def verify(self, variants):
        count = verify_files(self.root, self.references["files"])
        for variant in variants:
            require(variant in self.layouts, "Unknown variant: " + variant)
            count += verify_files(self.artifacts, self.layouts[variant]["files"])
            design = self.design(variant) / "GPUSimDesign.bin"
            with design.open("rb") as stream:
                magic = stream.read(8)
            wide = self.layouts[variant]["abi"] == "wide32"
            require(magic.startswith(b"TCWIDE1") == wide, "Model ABI differs: " + variant)
        return count

    def design(self, variant):
        return relative_file(self.artifacts, self.layouts[variant]["directory"])

    def contract(self, variant):
        return {"model_files": self.layouts[variant]["files"],
                "reference_manifest": identity(self.root / "reference_manifest.json"),
                "model_manifest": identity(self.root / "model_manifest.json")}

    def prepare(self, variants, output):
        self.verify(variants)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        files = {}
        for variant in variants:
            target = output / variant
            target.mkdir()
            text = bind_watches(self.design(variant), self.watches,
                                self.layouts[variant]["regpool_bytes"])
            (target / "nanhu-watch.tsv").write_text(text, encoding="utf-8")
            for source in (self.root / "workloads").iterdir():
                require(source.is_file(), "Unexpected workload directory")
                shutil.copyfile(source, target / source.name)
            for path in target.iterdir():
                files[path.relative_to(output).as_posix()] = identity(path)
        write_json(output / "prepared.json", {
            "schema_version": 1, "variants": list(variants), "files": files,
            "contracts": {variant: self.contract(variant) for variant in variants},
            "numeric_watch_offsets_reused": False})

    def verify_prepared(self, prepared, variants):
        prepared = Path(prepared)
        metadata = read_json(prepared / "prepared.json")
        verify_files(prepared, metadata["files"])
        for variant in variants:
            require(metadata["contracts"].get(variant) == self.contract(variant),
                    "Prepared contract differs: " + variant)
            expected = bind_watches(self.design(variant), self.watches,
                                    self.layouts[variant]["regpool_bytes"])
            require((prepared / variant / "nanhu-watch.tsv").read_text() == expected,
                    "Watch binding differs: " + variant)
        return metadata


def assemble(package, runtime_root, output):
    """Copy a complete movable suite from normalized artifacts and runtime source."""
    variants = list(package.layouts)
    package.verify(variants)
    runtime = Path(runtime_root).resolve()
    output = Path(output).resolve()
    require(runtime.is_dir(), "Missing runtime source directory")
    require(not output.exists(), "Package destination already exists")
    require(not output.is_relative_to(package.root) and not output.is_relative_to(runtime),
            "Package destination cannot be inside its source tree")
    output.mkdir(parents=True)
    for source in package.root.iterdir():
        if source.suffix == ".py" or source.name in (
                "README.md", ".gitignore", "model_manifest.json", "reference_manifest.json",
                "endpoints.json", "watches.json", "references", "workloads"):
            if source.is_dir():
                shutil.copytree(source, output / source.name)
            else:
                shutil.copyfile(source, output / source.name)
    for variant in variants:
        for name in package.layouts[variant]["files"]:
            target = relative_file(output, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(relative_file(package.artifacts, name), target)
    shutil.copytree(runtime, output / "runtime", ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "build", "out", "*.o", "*.a", "*.so"))
    Package(output).verify(variants)

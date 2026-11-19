#!/usr/bin/env python3
"""Verify, build, check, benchmark, and profile the frozen Nanhu layouts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from common import paired_summary, require, write_json
from execution import Runtime, checked_gate, command, environment, run_action
from package import Package


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact-root", type=Path, help="Root containing models/; defaults to this directory")
    commands = result.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="CPU-only payload SHA256 verification")
    verify.add_argument("--variants", nargs="+", default=None)
    prepare = commands.add_parser("prepare", help="CPU-only layout-specific watch binding")
    prepare.add_argument("--variants", nargs="+", default=None)
    prepare.add_argument("--output", type=Path, default=Path("prepared"))
    build = commands.add_parser("build", help="Build both observation paths; launches no GPU work")
    build.add_argument("--runtime-root", type=Path, required=True)
    build.add_argument("--build-root", type=Path, default=Path("build"))
    build.add_argument("--arch", choices=("sm_86", "sm_120"), default="sm_86")
    build.add_argument("--abis", nargs="+", choices=("wide32", "legacy16"), default=["wide32", "legacy16"])
    build.add_argument("--nvcc", default="nvcc")
    build.add_argument("--dry-run", action="store_true")
    for name in ("check", "bench", "profile"):
        action = commands.add_parser(name)
        action.add_argument("--prepared-root", type=Path, default=Path("prepared"))
        action.add_argument("--build-root", type=Path, default=Path("build"))
        action.add_argument("--output", type=Path, required=True)
        action.add_argument("--timeout", type=float, default=1800)
        action.add_argument("--dry-run", action="store_true", help="Print exact commands without launching GPU work")
        if name != "check":
            action.add_argument("--checked-root", type=Path, required=True)
        if name != "profile":
            action.add_argument("--variants", nargs="+", default=None)
        else:
            action.add_argument("--variant", default="p40")
            action.add_argument("--tool", choices=("nsys", "ncu"), required=True)
            action.add_argument("--profiler", help="Profiler executable; defaults to --tool")
            action.add_argument("--profile-arg", action="append", default=[],
                                help="Additional profiler argument; use --profile-arg=--flag")
        if name == "bench":
            action.add_argument("--baseline", default="p20")
            action.add_argument("--pairs", type=int, default=3)
    return result


def plan_checks(variants):
    plan = []
    for variant in variants:
        plan.append({"label": "resources-" + variant, "kind": "resources", "variant": variant})
        for loops in (1, 1000):
            plan.append({"label": f"checked-{variant}-loop{loops}", "kind": "checked",
                         "variant": variant, "loops": loops})
    return plan


def plan_bench(variants, baseline, pairs):
    require(pairs >= 3, "At least three paired samples required")
    candidates = [v for v in variants if v != baseline]
    require(candidates and len(candidates) == len(set(candidates)), "Expected unique candidates")
    plan = [{"label": "warmup-" + v, "kind": "warmup", "variant": v, "loops": 1000}
            for v in dict.fromkeys([baseline, *candidates])]
    for candidate in candidates:
        for index in range(1, pairs + 1):
            order = (baseline, candidate) if index % 2 else (candidate, baseline)
            for variant in order:
                plan.append({"label": f"{candidate}-pair{index}-{variant}", "kind": "timing",
                             "variant": variant, "loops": 1000, "candidate": candidate, "pair": index})
    return plan


def run_plan(package, args, plan):
    variants = list(dict.fromkeys(item["variant"] for item in plan))
    package.verify(variants)
    package.verify_prepared(args.prepared_root, variants)
    runtimes = {v: Runtime(args.build_root, package.layouts[v]["abi"]) for v in variants}
    commands = []
    for item in plan:
        runtime = runtimes[item["variant"]]
        directory = args.output.resolve() / item["label"]
        if item["kind"] == "resources":
            app = [str(runtime.binary("resources")), str(package.design(item["variant"]))]
        else:
            app = command(package, args.prepared_root, runtime, item["variant"], item["loops"],
                          directory, item["kind"] == "checked")
        if item["kind"] == "profile":
            profiler = args.profiler or args.tool
            if args.tool == "nsys":
                prefix = [profiler, "profile", "--trace=cuda,nvtx,osrt", "--output", str(directory / "nsys")]
            else:
                prefix = [profiler, "--set", "basic", "--replay-mode", "application",
                          "--target-processes", "all", "--export", str(directory / "ncu")]
            app = [*prefix, *args.profile_arg, *app]
        commands.append({**item, "command": app})
    if args.dry_run:
        print(json.dumps({"GPU_executed": False, "plan": commands}, indent=2))
        return 0
    env = environment()
    require(not args.output.exists(), "Action output directory already exists")
    if args.command != "check":
        for variant in variants:
            checked_gate(package, args.checked_root, runtimes[variant], variant, env)
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", commands)
    write_json(args.output / "environment.json", env)
    rows, failures = [], {}
    for item in commands:
        variant = item["variant"]
        if variant in failures or item.get("candidate") in failures:
            continue
        try:
            receipt = run_action(package, runtimes[variant], variant, item.get("loops"), item["kind"],
                                 item["command"], args.output / item["label"], env, args.timeout)
            rows.append({**item, "result": receipt["result"], "process_seconds": receipt["process_seconds"],
                         "receipt": item["label"] + "/action.json"})
        except (OSError, ValueError, subprocess.SubprocessError, StopIteration) as error:
            failures[variant] = {"action": item["label"], "error": str(error)}
            if args.command != "check" and variant == getattr(args, "baseline", None):
                break
        write_json(args.output / "progress.json", {"rows": rows, "failures": failures})
    summary = {"status": "failed" if failures else "passed", "rows": rows, "failures": failures,
               "warmups_excluded": sum(row["kind"] == "warmup" for row in rows),
               "profiled_samples_used_for_speed": False}
    if args.command == "bench":
        summary["candidates"] = {}
        for candidate in dict.fromkeys(item["candidate"] for item in plan if "candidate" in item):
            samples = [row for row in rows if row.get("candidate") == candidate]
            if candidate not in failures and args.baseline not in failures:
                summary["candidates"][candidate] = paired_summary(samples, args.baseline, candidate)
    package.verify(variants)
    package.verify_prepared(args.prepared_root, variants)
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "summary": str(args.output / "summary.json")}, indent=2))
    return bool(failures)


def main():
    args = parser().parse_args()
    root = Path(__file__).resolve().parent
    if args.command == "build":
        for abi in args.abis:
            app = [sys.executable, str(args.runtime_root.resolve() / "build_nanhu.py"),
                   "--abi", abi, "--arch", args.arch, "--output", str(args.build_root.resolve() / abi),
                   "--nvcc", args.nvcc]
            if args.dry_run:
                app.append("--dry-run")
            subprocess.run(app, check=True)
        return 0
    package = Package(root, args.artifact_root)
    if args.command == "verify":
        print(f"Verified {package.verify(args.variants or list(package.layouts))} payloads.")
        return 0
    if args.command == "prepare":
        package.prepare(args.variants or list(package.layouts), args.output)
        print("Prepared all requested symbol bindings: " + str(args.output))
        return 0
    if args.command == "check":
        plan = plan_checks(args.variants or list(package.layouts))
    elif args.command == "bench":
        plan = plan_bench(args.variants or list(package.layouts), args.baseline, args.pairs)
    else:
        plan = [{"label": "profile-" + args.variant, "kind": "profile", "variant": args.variant, "loops": 1000}]
    return run_plan(package, args, plan)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        sys.exit("Nanhu harness: " + str(error))

# Ready-to-run Nanhu partitions

This directory is a portable experiment suite for the accepted Nanhu v5
single-clock model. Prepartitioned models are Git LFS files in `models/`;
they are repository content, not GitHub Release assets. The fork's `.lfsconfig`
uses `zhangziqing/toucan-nanhu-models` as storage because GitHub public forks
cannot accept new LFS uploads. `git lfs pull` resolves these pointers directly.
No compiler rerun,
original source-host directory, or Python package installation is needed to
execute a model. Python 3.9 or newer and the matching `toucanGPURT` source
are required. Building the runner requires NVCC with C++20 support.

| Layout | ABI | partSize | Round partitions | Actual maximum local bytes |
| --- | --- | ---: | --- | ---: |
| p10 | legacy16 | 0.10 | 944 + 8,583 | 59,983 |
| p20 | wide32 / TCWIDE1 | 0.20 | 472 + 2,146 | 75,968 |
| p25 | wide32 / TCWIDE1 | 0.25 | 378 + 1,375 | 82,941 |
| p30 | wide32 / TCWIDE1 | 0.30 | 315 + 955 | 87,885 |
| p40 | wide32 / TCWIDE1 | 0.40 | 236 + 537 | 94,269 |

The frozen wide pool budget is 98,304 bytes; the legacy budget is 65,534.
All layouts have exactly two rounds, cut target 0.70, no native repartitions,
and no capacity repair. The compiler and flattened-input SHA256 identities,
each payload's size and SHA256, and portable paths are in
`model_manifest.json`. `p20` is the paired sweep baseline. Comparing `p10`
and `p20` changes both ABI and partitioning.

The [RTX 3090 retest and profiling report](RTX3090.md) records all five
passing layouts, fresh paired timings and the retained NSYS p40 cleanup
failure. The best tested layout, p40, reached 1.502561x paired evaluation
speedup over p20; all raw timing rows are included with the report.

## Fetch and build

Run these commands from this directory after `git submodule update --init runtime`
at the repository root. The exact runtime commit is pinned by that submodule.

```sh
git lfs install
git lfs pull
python3 nanhu.py verify
python3 nanhu.py prepare --output prepared
python3 nanhu.py build --runtime-root ../../runtime --arch sm_86 --build-root build
```

`sm_86` selects the RTX 3090 target. `--nvcc /path/to/nvcc` selects a CUDA
toolchain explicitly; `--arch sm_120` builds for the previously used RTX PRO
6000 Blackwell. The architecture flag does not assert GPU compatibility:
the resource action queries the current GPU using the actual model's shared
allocation and verifies cooperative residency. No value-pool spill is added.
Build cost and tool versions are recorded separately in each ABI's
`build.json` and command logs. The checked and batch-final paths are both
built, with 64-register cap, `-O3`, C++20, and `-lineinfo`.

`build` launches no GPU evaluation. Add `--dry-run` to print compiler commands.
Model verification is CPU-only and fails if LFS pointer text is present.
Preparation streams the symbol map, checks 117 required IO bindings, and
generates six watch bindings by symbol name for each layout. It never copies
numeric watch offsets between layouts. Prepared files contain portable paths
and remain valid after moving the complete suite.

## Check, benchmark, profile

All actual GPU commands must run under the designated external GPU scheduler
and its exclusive allocation. This suite does not acquire a second lock,
choose a GPU, or alter scheduler state. The scheduler supplies
`CUDA_VISIBLE_DEVICES` and CPU affinity and supervises the complete process
tree. Keep those settings identical across the check, bench, and profile
actions. On a scheduler-free private host, the owner must first ensure the
GPU is exclusively available.

```sh
python3 nanhu.py check --prepared-root prepared --build-root build --output runs/check
python3 nanhu.py bench --prepared-root prepared --build-root build \
  --checked-root runs/check --baseline p20 --variants p10 p25 p30 p40 \
  --pairs 3 --output runs/bench
python3 nanhu.py profile --prepared-root prepared --build-root build \
  --checked-root runs/check --variant p40 --tool nsys --output runs/nsys-p40
python3 nanhu.py profile --prepared-root prepared --build-root build \
  --checked-root runs/check --variant p40 --tool ncu --output runs/ncu-p40
```

Choose a fresh output directory for every invocation. `--variants` limits
`verify`, `prepare`, `check`, or `bench`; `--variant` selects one profile
layout. Use the same preparation directory for checks and timing. The global
`--artifact-root DIRECTORY` option accepts an external normalized root
containing `models/{p10,p20,p25,p30,p40}/`.

Before checking each layout, the resource probe records device identity,
static and dynamic shared bytes, registers, occupancy, threads, cooperative
grid and selected batch kernel. A failed resource or checked action prevents
further actions for that layout. Correctness checks run both loop1 and
loop1000 with full commit/device logs and 169 sampled watch records. All 32
final GPRs, trap, retirement, device counters and cycle endpoints must match.
First differences are reported by field, text line, or watch-record byte.

| Workload | Total cycles | Trap cycle | Retired | RAM reads |
| --- | ---: | ---: | ---: | ---: |
| loop1 | 8,511 | 8,507 | 19 | 120 |
| loop1000 | 10,645 | 10,641 | 8,011 | 108 |

Both use 64 reset cycles, first commit at 631, zero RAM writes, and three
flash reads. Complete reference files are deterministic gzip payloads under
`references/`, with SHA256 identities in `reference_manifest.json`. Their
commit traces match the original RTL reference. PC and instruction columns
are oracle values, not observed DUT PC/instruction. The six watches cover
only specified sampled states. Raw state pools are not compared across
layouts; this workload validation is not a general equivalence proof.

`bench` requires both prior checked actions on the same GPU, driver, visible
device selection, CPU affinity, model inputs, and built runtime. It revalidates
those receipts and all trace hashes. One unprofiled warmup per layout is
excluded. Each candidate gets at least three fresh pairs against `p20`,
alternating baseline/candidate then candidate/baseline. Every timing sample
checks final state and counters; it collects no per-cycle trace.

`summary.json` retains every raw evaluation, loop wall, process wall, cycles/s
and recomputed pair ratio. The median of pair ratios is reported, rather than
a ratio of medians or a comparison with old GPU results. Evaluation includes
the fixed reset and persistent-kernel synchronization; loop wall also
includes final compact observation. Initialization and terminal full-pool
diagnostic copies are excluded from evaluation/loop wall and included in
process wall. Throughput uses 10,645 divided by loop wall.

Profiling always uses separate output and never contributes to benchmark
speedup. NSYS defaults to CUDA/NVTX/OS-runtime tracing. NCU defaults to the
`basic` set and **application replay** (`--replay-mode application`), because
this workload uses cooperative persistent execution and GPU-resident state.
Each replay must start from identical workload input and produce deterministic
kernel launches; retain every report and any unsupported-counter warning.
Use `--profile-arg=--section` followed by another `--profile-arg=SECTION` to
request an available section. For example, add
`--profile-arg=--section --profile-arg=LaunchStats`. Profiler versions and
available sections can differ between machines. A profiler rejection is a
failed profile action, not a passing benchmark. Use `--profiler PATH` for
an explicit NSYS/NCU binary. Add `--dry-run` to `check`, `bench`, or `profile`
to inspect exact commands without launching them.

## Packaging and maintenance

The runtime source's `build_nanhu.py` contract is:

```sh
python3 RUNTIME/build_nanhu.py --abi wide32 --arch sm_86 --output BUILD/wide32 --nvcc nvcc
python3 RUNTIME/build_nanhu.py --abi legacy16 --arch sm_86 --output BUILD/legacy16 --nvcc nvcc
```

Each output's `build.json.binaries` uses relative paths:
`checked/nanhu-checked`, `final/nanhu-final`, and `final/resource-probe`.
The raw checked runner accepts
`DESIGN IMAGE OUTPUT MAX_CYCLES LOOP_COUNT`; the final runner additionally
accepts `batch-final`. The design filename is `GPUSimDesign.bin` for every
layout; wide models retain the `TCWIDE1` header despite this portable alias.

To assemble a movable directory from external artifacts and runtime source:

```sh
python3 prepare_package.py --artifact-root MODEL_ROOT --runtime-root RUNTIME_SOURCE --output PACKAGE_DIR
```

The artifact root must match the normalized `models/` paths in the manifest.
The output contains scripts, references, workloads, all model payloads and a
runtime source copy, without build products. A regular Git checkout is not
required by the Python harness. Record any runtime source revision separately
when preparing an archive rather than a checkout.

`import_accepted.py` is a maintainer-only importer for the two original
accepted evidence archives. It verifies archive/member hashes, copies only
small references and workloads, and freezes model identities; it does not
copy large models. `prepare_package.py` verifies every normalized payload
before copying. Neither script uploads files or creates a Release.

```sh
python3 -m unittest -v
```

CPU tests cover manifest/binary/trace tampering, path escape, symbol sections,
missing pairs, pair recomputation, standalone package assembly and a synthetic
checked-to-bench flow. They do not establish GPU correctness, occupancy, or
performance. New RTX 3090 measurements must come from the scheduler's actual
check/bench/profile receipts; historical Blackwell results are not substituted.

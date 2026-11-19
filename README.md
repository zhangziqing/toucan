# Toucan

## Nanhu Ready-to-Run Branch

This fork's `nanhu-ready-v1` branch preserves the accepted Nanhu experiment:
strictly two original RepCut rounds and the versioned `TCWIDE1` wide local-pool
ABI. It starts at upstream `4dda518`; upstream `main` is retained separately.
The compiler in this branch emits wide32 designs. The companion runtime also
supports the frozen legacy16 baseline, selected explicitly at build time.

Five prepartitioned layouts (`partSize=0.10/0.20/0.25/0.30/0.40`), symbols,
workloads, hash manifests, reference traces and a portable test harness are in
[benchmarks/nanhu](benchmarks/nanhu/README.md). Models are Git LFS objects in
the repository, not Release assets. No RTL compiler build is required to run
these models. Install Git LFS, Python 3.10+ and a C++20-capable CUDA toolkit.
Because GitHub rejects new LFS uploads to public forks, `.lfsconfig` selects
the companion `zhangziqing/toucan-nanhu-models` repository's storage endpoint.
The fork keeps the model pointers and exact hashes at the paths used below.

```sh
git clone -b nanhu-ready-v1 https://github.com/zhangziqing/toucan.git
cd toucan
git lfs pull
git submodule update --init runtime
cd benchmarks/nanhu
python3 nanhu.py verify
python3 nanhu.py prepare --output prepared
python3 nanhu.py build --runtime-root ../../runtime --arch sm_86 --build-root build
```

The runtime is pinned as a submodule; `sm_86` selects RTX 3090 and `sm_120`
selects RTX PRO 6000 Blackwell with a supporting toolkit. GPU runs require an
exclusive scheduler allocation and at least 16 GiB free for the device RAM
model plus design/runtime overhead. Follow the suite's checked acceptance,
paired benchmark and separate profiling commands. New hardware results must
be validated; the historical Blackwell results do not validate a new build.

The [validated RTX 3090 sweep](benchmarks/nanhu/RTX3090.md) passed all five
layouts and measured 1.502561x paired evaluation speedup for p40 versus p20.
It includes raw timing samples, NCU replay checks, NSYS findings and the
retained p40 NSYS process-cleanup failure.

To build the compiler itself with an existing LLVM/MLIR/CIRCT installation:

```sh
cmake -S . -B build/nanhu -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=MLIR_CMAKE_DIR -DCIRCT_DIR=CIRCT_CMAKE_DIR -DLLVM_DIR=LLVM_CMAKE_DIR
cmake --build build/nanhu --target toucan check-design-graph \
  check-local-value-allocator check-micro-part-merging check-wide-shift check-array-index -j8
ctest --test-dir build/nanhu --output-on-failure
```

Put the built partitioners (`rcp`, `micro_partitioner`) on `PATH` for compiler
controls and recompilation. Local validation used Clang 19, CIRCT
`e272897aa3345c0c081551b8433844ce1df3cf82` and LLVM
`040a641988f6ed6f4fab250706ca2b620c1de2d8`. This newer CIRCT installation was
supplied explicitly, rather than the original upstream CIRCT submodule.
Compiler checks passed for allocation, graph construction, micro-part merging,
772,332 shift comparisons, 23,856 array-index comparisons and strict capacity
rejection. Two clock controls also passed 1,800 root cycles each against
Verilator and an independent state/memory oracle. Normalization remains opt-in
and limited to the documented Nanhu single-clock contract.

Toucan is a **GPU-accelerated RTL simulator** that compiles hardware designs described in FIRRTL/Verilog into efficient GPU simulation kernels.

Toucan lowers CIRCT's HW/Comb/Seq/SV dialects through a custom MLIR dialect pipeline, partitions the design (RepCut + micro-partitioner), and generates CUDA simulation binary.

## Paper & Related Repositories

If you use Toucan in your research, please cite:

> (*placeholder — paper citation will be added upon publication*)

For paper data reproduce, please check [this repo](https://github.com/ucsc-vama/toucan-demo-pure-gpu).

Toucan itself is a compiler, the interpreter running on GPU is [this repo](https://github.com/ucsc-vama/toucanGPURT).

## Prerequisites

| Dependency | Version |
|-----------|---------|
| CMake | >= 3.26 |
| Boost | (graph, filesystem) |
| MtKaHyPar | requires `tbb`, `hwloc` |
| toucan-micro-partitioner | (see `external/toucan-micro-partitioner`) |
| LLVM/MLIR | (built via CIRCT) |
| CIRCT | (see `external/circt`) |

## Building

Toucan uses Git submodules for its external dependencies:

```bash
git submodule update --init --recursive
```

Build all dependencies (LLVM/MLIR, CIRCT, partitioners):

```bash
cd external && ./build-deps.sh
```

Then build Toucan itself:

```bash
cmake --preset release
cmake --build --preset release -j$(nproc)
```

For a debug build with symbols and assertions, use the isolated debug build
tree instead:

```bash
cmake --preset debug
cmake --build --preset debug -j$(nproc)
```

The resulting executables are `build/release/tools/toucan/toucan` and
`build/debug/tools/toucan/toucan`. To configure manually, use separate build
directories with `-DCMAKE_BUILD_TYPE=Release` or `-DCMAKE_BUILD_TYPE=Debug`.

## Pipeline Overview

```
FIRRTL/Verilog
    ↓ (firtool)
CIRCT CoreDialect (HW, Comb, Seq, SV)
    ↓ ToucanHigh  — Remove SV/OM, split mem ports, split registers, expand delays
    ↓ Toucan4B    — Lower to 4-bit wide signals (registers, memories, comb ops, vectors)
    ↓ ToucanFlattened — Flatten hierarchy, dedup registers, merge constants
    ↓ Code Generation — Partition, micro-partition, allocate values, serialize .bin design
GPU Simulation
```

## Usage

GPU codegen shells out to the partitioner binaries (`rcp`, `micro_partitioner`, `MtKaHyPar`), which are installed into `install/bin`. Make sure they are on `PATH`:

```bash
export PATH="$PWD/install/bin:$PATH"
```

```bash
# Compile MLIR (CIRCT core dialect) to GPU simulation design
./build/release/tools/toucan/toucan input.mlir -o ./output_dir -gpu

# Specify input level and dump intermediate MLIR
./build/release/tools/toucan/toucan input.mlir -o ./output_dir -gpu --inputLevel=core --dump

# Tune partition size (ratio 0.01-10.0, default 1.0)
./build/release/tools/toucan/toucan input.mlir -o ./output_dir -gpu --partSize 0.5

# Remove SV print/assert ops (no I/O during sim)
./build/release/tools/toucan/toucan input.mlir -o ./output_dir -gpu --removeSVDialect
```

**Input levels:** `core` (default), `toucanHigh`, `toucan4B`, `toucanFlattened`

**Output files** (in `-o` directory):
- `GPUSimDesign.bin` — Compiled GPU simulation design
- `GPUSimSymbols.bin` — Symbol table
- `GPUSimIOSymbols.bin` — I/O symbol table
- `output.mlir` — (with `--dump`) Intermediate MLIR dump

The compiled design is run by the [Toucan GPU runtime](https://github.com/ucsc-vama/toucanGPURT).

## Directory Layout

- `include/toucan/` — Public headers and MLIR TableGen dialect definitions
- `lib/` — Library source (dialect, analysis, passes, simulation serialization)
- `tools/` — Executables (`toucan`, `firtool_test`)
- `external/` — Git submodules and build scripts for dependencies

## License

BSD 3-Clause. See `LICENSE`.

## Copy-bytecode regression check

The host-only checker exercises all 16 nibble values through copy-only
micro-partitions in a compiled design. It detects lane selectors accidentally
encoded as constants by `scheduleNOPMicroPart`.

```bash
g++ -std=c++20 -O2 -Iinclude -Iinclude/ToucanGPUSim \
  tests/check_nop_bytecode.cpp lib/ToucanGPUSim/Serializer.cpp \
  -o /tmp/check_nop_bytecode
/tmp/check_nop_bytecode ../rocket-gpu/designs/rocket21-1c/rocket21-1c.bin
```

This checks copy encoding; it does not replace end-to-end GPU execution tests.

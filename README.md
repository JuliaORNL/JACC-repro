# JACC-repro
Reproducibility information for JACC performance tests
- **GPU Test**:
    - Running with different problem sizes and GPU block dimensions
    - Metrics: Bandwidth (GB/s)
    - Machines tested: MI100 GPUs on Defiant, MI250x on Frontier (Odo)   
- **CPU Test**:
    - Running with different problem sizes and the number of threads running in parallel
        -  `OMP_NUM_THREADS` is set to: 128, 64, 32, 16, 8, 4, 2, 1
    - Metrics: Wall-clock runtime (ms)   
    - Machines tested: x86-based 64 cores CPU on Frontier

## Overview
GPU parameters:
- `Bx`, `By`, `Bz`: block dimension
    - total number of blocks = `Bx`\*`By`\*`Bz`
    - Maximum number of blocks:
        - MI100 (AMD): 1024
        - A100 (NVIDIA): 1024
-  `nx`, `ny`, `nz`: problem dimension
    - problem size = `nx`\*`ny`\*`nz`\*`sizeof(precision)`
    - Maximum problem size = GPU global memory size
        - Default `precision`: double

## Run Performance Test
To run performance test, `cd` into Performance-test folder and run the following commands (change paths to the ones on your machine):
### GPU test
```
chmod +x gpu-test.sh
./gpu-test.sh [kernel] [L] [Bx] [By] [Bz] [output].txt
```
- `kernel`: path to the compiled executable 
- `L`: problem dimension, only testing `nx`=`ny`=`nz`
    - Maximum `L`:
        - MI100 (AMD): 1024
        - A100 (AMD): 1024
- `Bx`, `By`, `Bz`: block dimension
- `output`: output file name to store test results
### CPU test
```
chmod +x cpu-test.sh
./cpu-test.sh [L] #run on current node
sbatch --threads-per-core=2 job.sl [L] #submit job to run on compute node
```
- `L`: problem dimension, only testing `nx`=`ny`=`nz`
    - Maximum `L`:
        - MI100 (AMD): 1024
        - A100 (AMD): 1024

## Test Results
test-results-AMD-gpu.csv: Results of running the 7-point stencil kernel on AMD GPUs (Mi100 and Mi250X) with different problem sizes and GPU block dimensions
- Column label: [GPU type]\_[L]\_[Bx]-[By]-[Bz]
- Each column: 100 runs for each configuration
- Data: bandwidth (GB/s)

## Profiling
### AMD GPUs
Profile using `rocprof` \
HIP:
```
rocprof --hsa-trace --stats -o prof_result.txt -i input.txt <executable>
```
```
# input.txt
pmc : FetchSize WriteSize 
pmc : TCC_HIT[0], TCC_MISS[0]
kernel: laplacian_kerne
```
JACC:
```
rocprof --stats -o profiling/prof_result.csv -i profiling/input.txt julia --project gray-scott.jl examples/settings-files.json 
```
```
# tracing
ENABLE_JITPROFILING=1 rocprofv2 --plugin perfetto --sys-trace --kernel-trace -o out julia --project gray-scott.jl examples/settings-files.json
```

## LLVM-IR
### AMD GPUs
Compile with `-S` flag
### JACC
Import `InteractiveUtils` in script and add `InteractiveUtils.@code_llvm` before `parallel_for()`.
# JACC-repro
Reproducibility information for JACC performance tests
- **GPU Test**:
    - Running with different problem sizes and GPU block dimensions
    - Metrics: Bandwidth (GB/s)   
- **CPU Test**:
    - Running with different problem sizes and the number of threads running in parallel
        -  `OMP_NUM_THREADS` is set to: 128, 64, 32, 16, 8, 4, 2, 1
    - Metrics: Wall-clock runtime (ms)   

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
To run performance test, `cd` into Performance-test folder and run the following commands:
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
./cpu-test.sh [kernel] [L]
```
- `kernel`: path to the compiled executable 
- `L`: problem dimension, only testing `nx`=`ny`=`nz`
    - Maximum `L`:
        - MI100 (AMD): 1024
        - A100 (AMD): 1024

## Test Results
test-results-AMD-gpu.csv: Results of running the 7-point stencil kernel on AMD GPUs (Mi100 and Mi250X) with different problem sizes and GPU block dimensions
- Column label: [GPU type]\_[L]\_[Bx]-[By]-[Bz]
- Each column: 100 runs for each configuration
- Data: bandwidth (GB/s)
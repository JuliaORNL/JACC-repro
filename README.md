# JACC-repro
Reproducibility information for JACC performance tests

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
```
chmod +x runtest.sh
./runtest.sh [kernel] [L] [output].txt
```
- `kernel`: path to the compiled executable 
- `output`: output file name to store test results
- `L`: problem dimension, only testing `nx`=`ny`=`nz`
    - Maximum `L`:
        - MI100 (AMD): 1024
        - A100 (AMD): 1024

## Test Results
test-results-AMD-gpu.csv: Results of running the 7-point stencil kernel on AMD GPUs (Mi100 and Mi250X) with different problem sizes and GPU block dimensions
- Column label: [GPU type]\_[L]\_[Bx]-[By]-[Bz]
- Each column: 100 runs for each configuration
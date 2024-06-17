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
./runtest.sh [kernel] [output].txt [N]  
```
- `kernel`: path to the compiled executable 
- `output`: output file name to store test results
- `N`: problem dimension, only testing `nx`=`ny`=`nz`
    - Maximum `N`:
        - MI100 (AMD): 1024
        - A100 (AMD): 1024


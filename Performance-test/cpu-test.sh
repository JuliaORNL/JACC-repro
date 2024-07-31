#!/bin/bash
# test wallclock runtime when running with different problem size and number of threads

PROGRAM="/JACC-repro/laplacian-openmp-amdclang++" # compiled script
L=$1 # problem size is L cubed
OUTPUT="results/result_$L.txt"

> $OUTPUT

for ((i=128; i>=1; i=i/2)); do
	export OMP_PROC_BIND=true
        export OMP_NUM_THREADS=$i
	export OMP_PLACES=threads

	echo "Problem size: $L cubed, $OMP_NUM_THREADS threads"
	output=$(.$PROGRAM $L $L $L)

        number=$(echo "$output" | grep -oP 'kernel took: \K[0-9]+(\.[0-9]+)?')

        # store results like: num threads,runtime
        if [ -n "$number" ]; then
                echo "$i, $number" >> "$OUTPUT"
        fi
done

echo "Results stored"
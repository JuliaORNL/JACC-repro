#!/bin/bash

# test wallclock runtime when running with different problem size and number of threads
PROGRAM=$1 # compiled script
L=$2 # problem size is L cubed
OUTPUT="results_$L.txt"

> $OUTPUT

for ((i=128; i>=1; i=i/2)); do
        echo "Problem size: $L cubed, $i threads"
        output=$(OMP_NUM_THREADS=$i $PROGRAM $L $L $L)

        number=$(echo "$output" | grep -oP 'kernel took: \K[0-9]+(\.[0-9]+)?')

        # store results like: num threads,runtime
        if [ -n "$number" ]; then
                echo "$i,$number" >> "$OUTPUT"
        fi
done

echo "Results stored"
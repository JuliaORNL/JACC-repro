#!/bin/bash
# test wallclock runtime when running with different problem size and number of threads

GS_DIR="/ccsopen/home/y1e/jacc/GrayScott.jl" 
GS_EXE=$GS_DIR/gray-scott.jl
GS_IN=$GS_DIR/examples/settings-files.json

L=$1 # problem size is L cubed

for ((i=128; i>=1; i=i/2)); do
	OUTPUT="results/full_$L-$i.txt"
	
	> $OUTPUT

	export JULIA_NUM_THREADS=$i

	echo "Problem size: $L cubed, $JULIA_NUM_THREADS threads"
	julia --project=$GS_DIR $GS_EXE $GS_IN > $OUTPUT

    grep -oP 'kernel took: \K[0-9]+(\.[0-9]+)?' $OUTPUT > "results/result_$L-$i.txt"
done

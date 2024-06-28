#!/bin/bash

# test bandwidth when running with different problem size
PROGRAM=$1 # file path to compiled kernel
L=$2 
OUTPUT_FILE=$3

Bx=1024 # Maxium Bx * By * Mz = 1024
By=1
Bz=1

> $OUTPUT_FILE

# Function to print a progress bar
print_progress_bar() {
    local percent=$(( ($1 * 100) / $2 ))
    local done_chars=$(( ($percent * $3) / 100 ))
    local left_chars=$(( $3 - $done_chars ))

    printf "\rProgress: [%-${3}s] %d%%" \
           $(printf '#%.0s' $(seq $done_chars)) \
           $percent
}

echo "Processing input: $L"
results=()
for run in {1..1000}; do
        result=$($PROGRAM $L $L $L $Bx $By $Bz)
        bandwidth=$(echo "$result" | grep -oP 'effective memory bandwidth: \K[0-9]+(\.[0-9]+)?')

        if [ $? -eq 0 ]; then
                results+=($bandwidth)
        else
                echo "program failed for input $L on run $run."
                results+=("N/A")
        fi

        if (( run % 10 == 0 )); then
                print_progress_bar $run 1000 30
        fi
done

echo

results_str=$(IFS=,; echo "${results[*]}")
echo "$L,$results_str" >> $OUTPUT_FILE

echo "All results have been recorded in $OUTPUT_FILE"
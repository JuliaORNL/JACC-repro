#!/bin/bash

# test bandwidth when running with different problem size and block dimensions
PROGRAM=$1
L=$2
Bx=$3
By=$4
Bz=$5
OUTPUT_FILE="./block_test_results/results_$L($Bx,$By,$Bz).txt"

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
echo "Bx,By,Bz = $Bx,$By,$Bz"
results=()
for run in {1..100}; do
        result=$($PROGRAM $L $L $L $Bx $By $Bz)
        bandwidth=$(echo "$result" | grep -oP 'effective memory bandwidth: \K[0-9]+(\.[0-9]+)?')

        if [ $? -eq 0 ]; then
                results+=($bandwidth)
        else
                echo "program failed for input $L on run $run."
                results+=("N/A")
        fi

        if (( run % 10 == 0 )); then
                print_progress_bar $run 100 30
        fi
done

echo

results_str=$(IFS=,; echo "${results[*]}")
echo "$L,$results_str" >> $OUTPUT_FILE

echo "All results have been recorded in $OUTPUT_FILE"
#!/bin/bash

PROGRAM="./finite-difference/examples/laplacian_dp_kernel1"
OUTPUT_FILE="test_results.txt"

> $OUTPUT_FILE

MAX=1625
Bx=1024
By=1
Bz=1
TEST="b"

# Function to print a progress bar
print_progress_bar() {
    local percent=$(( ($1 * 100) / $2 ))
    local done_chars=$(( ($percent * $3) / 100 ))
    local left_chars=$(( $3 - $done_chars ))

    printf "\rProgress: [%-${3}s] %d%%" \
           $(printf '#%.0s' $(seq $done_chars)) \
           $percent
}

for ((n=32; n<=MAX; n*=2)); do
        echo "Processing input: $n"
        results=()

        for run in {1..1000}; do
                result=$($PROGRAM $n $n $n $Bx $By $Bz $TEST)

                if [ $? -eq 0 ]; then
                        results+=($result)
                else
                        echo "program failed for input $n on run $run."
                        results+=("N/A")
                fi

                if (( run % 10 == 0 )); then
                        print_progress_bar $run 1000 30
                fi
        done

        echo

        results_str=$(IFS=,; echo "${results[*]}")
        echo "$n,$results_str" >> $OUTPUT_FILE
done

echo "All results have been recorded in $OUTPUT_FILE"

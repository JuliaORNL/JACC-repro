#!/bin/bash

PROGRAM=$1 # file path to compiled kernel
L=$2 # problem size
Bx=$3 # block dims
By=$4
Bz=$5
ITER=1000
# TEST="b"
OUTPUT_FILE=$6
#EXEC=($PROGRAM $L $L $L $Bx $By $Bz)

> $OUTPUT_FILE

# Function to print a progress bar
function ProgressBar {
    # Process data
    let _progress=($1*100/$2*100)/100
    let _done=(${_progress}*4)/10
    let _left=40-$_done

    # Build progressbar string lengths
    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")

    # Build progressbar strings and print the ProgressBar line
    printf "\rProgress : [${_fill// /#}${_empty// /-}] ${_progress}%%"

}

echo "Processing for L = $L, block sizes $Bx, $By, $Bz"
results=()
for run in {1..$ITER}; do
        #result=$EXEC
        result=$($PROGRAM $N $N $N $Bx $By $Bz $TEST)

        if [ $? -eq 0 ]; then
                results+=($result)
        else
                echo "program failed on run $run."
                #results+=("N/A")
        fi

        if (( run % 10 == 0 )); then
                ProgressBar $run $ITER
        fi
done

echo

#results_str=$(IFS=,; echo "${results[*]}")
#echo "$N,$results_str" >> $OUTPUT_FILE

awk -v var="${results[*]}" 'print $9' > $OUTPUT_FILE


echo "All results have been recorded in $OUTPUT_FILE"
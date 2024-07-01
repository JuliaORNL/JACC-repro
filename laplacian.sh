#!/bin/bash

PROGRAM=$1 # file path to compiled kernel
L=$2 # problem size
Bx=$3 # block dims
By=$4
Bz=$5
ITER=$7 # iterations
TEST=$6 # a = print only bandwidth, b = print full message
OUTPUT_FILE=$8 # name of output file
#EXEC=($PROGRAM $L $L $L $Bx $By $Bz)

> $OUTPUT_FILE

# Function to print a progress bar
ProgressBar() {
    # Process data
    let _progress=($1*100/1000)
    let _done=(${_progress}*4)/10
    let _left=40-$_done

    # Build progressbar string lengths
    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")

    # Build progressbar strings and print the ProgressBar line
    printf "\rProgress : [${_fill// /#}${_empty// /-}] ${_progress}%%"

}

# Function to print a progress bar
print_progress_bar() {
    local percent=$(( ($1 * 100) / $2 ))
    local done_chars=$(( ($percent * 3) / 10 ))
    local left_chars=$(( 30 - $done_chars ))

    printf "\rProgress: [%-30s] %d%%" \
           $(printf '#%.0s' $(seq $done_chars)) \
           $percent
}

echo "Processing for L = $L, block sizes $Bx, $By, $Bz"
results=()
for (( i=1; i <= $ITER; ++i ))
do
	result=$($PROGRAM $L $L $L $Bx $By $Bz $TEST)

        if [ $? -eq 0 ]; then
                results+=($result)
        else
                echo "program failed on run $run."
                #results+=("N/A")
        fi

        #if (( $i % 10 == 0 )); then
        #       ProgressBar $i
	       print_progress_bar $i $ITER
        #fi
done

echo

printf "L = $L, block sizes $Bx, $By, $Bz\n" > $OUTPUT_FILE
printf "%s\n" "${results[@]}" > $OUTPUT_FILE

#awk -v var="${results[*]}" 'BEGIN { print $9 }' > $OUTPUT_FILE
#echo ${results[*]}

#for r in "${results[*]}" ; do
	#echo "r = " $r
#	awk -v var="$r" '{ split(var,arrayval," "); print(arrayval[9])}' > $OUTPUT_FILE
#done

echo "All results have been recorded in $OUTPUT_FILE"

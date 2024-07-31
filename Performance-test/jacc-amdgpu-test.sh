#!/bin/bash

GS_DIR=/home/y1e/GrayScott.jl
GS_EXE=$GS_DIR/gray-scott.jl
GS_IN=$GS_DIR/examples/settings-files.json
L=512

julia --project=$GS_DIR $GS_EXE $GS_IN > out-$L.txt
grep -oP 'effective memory bandwidth: \K[0-9]+(\.[0-9]+)?' out-$L.txt > result-$L.txt
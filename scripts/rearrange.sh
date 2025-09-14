#!/bin/bash

# Exit if no argument is given
if [ -z "$1" ]; then
  echo "Usage: $0 <folder_name>"
  exit 1
fi

FOLDER_NAME="$1"

# Find directories missing new_annotation.json
find ./frames/"$FOLDER_NAME"/ -type d ! -exec test -e "{}/new_annotation.json" \; -print > test.txt
sort test.txt > test_sort.txt
rm test.txt
tail -n +2 test_sort.txt > sorted_text.txt
rm test_sort.txt

# Input file containing paths
INPUT_FILE="sorted_text.txt"

# Extract numeric parts, remove leading zeros, and sort
numbers=$(sed 's|.*/||' "$INPUT_FILE" | sed 's/^0*//' | sort -n)

# Convert to compressed range format
compress_ranges() {
    local arr=($@)
    local start=${arr[0]}
    local end=$start
    local output=""

    for ((i=1; i<${#arr[@]}; i++)); do
        if [[ ${arr[i]} -eq $((end + 1)) ]]; then
            end=${arr[i]}
        else
            if [[ $start -eq $end ]]; then
                output+="$start,"
            else
                output+="$start-$end,"
            fi
            start=${arr[i]}
            end=$start
        fi
    done

    # Handle last range
    if [[ $start -eq $end ]]; then
        output+="$start"
    else
        output+="$start-$end"
    fi

    echo "$output"
}

compress_ranges $numbers
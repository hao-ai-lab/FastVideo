#!/bin/bash

# Define source and destination directories
SOURCE_DIR="/home/hal-weiz/hy15_tf_init_3333/prompts_16k"
DEST_DIR="/home/hal-weiz/hy15_tf_init_3333/prompts_16k_32_files"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

echo "Merging files from $SOURCE_DIR to $DEST_DIR..."

# Loop to create 32 merged files from 64 source files
for i in {1..32}; do
    # Calculate source file indices
    # i=1 -> uses 1 and 2
    # i=2 -> uses 3 and 4
    # ...
    # i=32 -> uses 63 and 64
    idx1=$(( (i-1)*2 + 1 ))
    idx2=$(( (i-1)*2 + 2 ))
    
    file1="${SOURCE_DIR}/v2m_${idx1}.txt"
    file2="${SOURCE_DIR}/v2m_${idx2}.txt"
    outfile="${DEST_DIR}/v2m_${i}.txt"
    
    if [[ -f "$file1" && -f "$file2" ]]; then
        # Concatenate the two files
        cat "$file1" "$file2" > "$outfile"
        echo "Created v2m_${i}.txt from v2m_${idx1}.txt and v2m_${idx2}.txt"
    else
        echo "Warning: Missing source files for v2m_${i}.txt (expected v2m_${idx1}.txt and v2m_${idx2}.txt)"
    fi
done

echo "Done. Created 32 merged files in $DEST_DIR"

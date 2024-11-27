#!/bin/bash

# Validate input
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_directory>"
    exit 1
fi

root_dir="$1"

# Check for available text extraction tools
if command -v lynx &> /dev/null; then
    # Find and process HTML files
    find "$root_dir" -type f -name "*.html" | while read -r file; do
        # Generate output filename in the same directory
        output_file="${file%.html}.txt"
        
        # Process each HTML file
        echo "Processing $file"
        
        # Attempt text extraction, handle potential errors
        if lynx -dump -nolist "$file" > "$output_file" 2>/dev/null; then
            echo "Extracted: $output_file"
        else
            echo "Error extracting text from $file"
        fi
    done
    
    echo "Text extraction complete."
else
    echo "Error: Missing dependency: Lynx. Please install lynx package."
    exit 1
fi

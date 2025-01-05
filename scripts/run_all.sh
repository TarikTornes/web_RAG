#!/bin/bash

cd "$(dirname "$0")" || exit 1

# Define file paths
DATA_DIR="../data"
CHUNKS_FILE="${DATA_DIR}/chunks.pkl"
EMBEDDINGS_FILE="${DATA_DIR}/embeddings.pkl"

# Ensure the data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "The 'data/' directory does not exist. Please create it and add the required files."
    exit 1
fi

# Check if the required files exist
if [[ -f "$CHUNKS_FILE" && -f "$EMBEDDINGS_FILE" ]]; then
    echo "Both 'chunks.pkl' and 'embeddings.pkl' are found. Starting the RAG-System"
    cd ..
    python3 -m src.main  # Replace with your actual script name
    cd scripts
else
    echo "One or both required files are missing."
    echo "Do you want to extract web files and download them locally? [Y/n]"
    read -r user_input

    if [[ "$user_input" =~ ^[Yy]$ ]]; then
        echo "Running the script to chunk and embed the webpages..."
        cd ..
        python3 -m src.chunking.chunking_main # Replace with your actual script name
        python3 -m src.embedding.embedding_main # Replace with your actual script name
        cd scripts
    else
        echo "Please perform webpage retrieval and place the 'chunks.pkl' and 'embeddings.pkl' files in the 'data/' directory."
        exit 1
    fi
fi

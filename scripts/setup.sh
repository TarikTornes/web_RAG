#!/bin/bash

cd "$(dirname "$0")" || exit 1

# Checks if the directory data/webpages exists
if [ ! -d "../data/webpages" ]; then
  # If it doesn't exist, create the directory
  mkdir -p "../data/webpages"
  echo "Directory 'data/webpages' has been created."
fi

echo "Do you want to download the webpages? [Y/n]"
echo "If you already have ensure that they are in directory data/webpages/ and enter <n> !"
read -r user_input

if [[ "$user_input" =~ ^[Yy]$ ]]; then
    echo "URL of the website to be downloaded (e.g. https://www.uni.lu):"
    read -r website_url
    echo "Webpages will be downloaded now ..."
    sleep 2
    # echo | ls

    if ./webpage_retrieval.sh "$website_url"; then
        echo "Webpages were successfully downloaded!"
        sleep 1
    else
        echo "Error: Failed to download the webpages!"
        exit 1
    fi
else
    echo "Skipping webpage download."
fi

echo "Content of downloaded webpages will extracted ..."
sleep 1

if ./extract_webtext.sh ../data/webpages; then
    echo "Content extraction successfully completed!"
else
    echo "Error: Failed to extract content of webpages!"
    exit 1
fi

echo "Setup is finished!"


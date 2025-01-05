#!/bin/bash

cd "$(dirname "$0")" || exit 1

echo "Did you already download the webpages? [Y/n]"
read -r user_input

if [[ "$user_input" =~ ^[Yy]$ ]]; then
    echo "URL of the website to be downloaded (e.g. https://www.uni.lu):"
    read -r website_url
    echo "Webpages will be downloaded now ..."
    sleep 2

    if ./scripts/webpage_retrieval "$website_url"; then
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

if ./scripts/extract_webtext ../data/webpages; then
    echo "Content extraction successfully completed!"
else
    echo "Error: Failed to extract content of webpages!"
    exit 1
fi

echo "Setup is finished!"


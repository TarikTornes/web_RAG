#!/bin/bash

cd "$(dirname "$0")" || exit 1

# Check if a URL is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <website_url>"
    exit 1
fi

# Extract domain from the provided URL
DOMAIN=$(echo "$1" | sed -e 's|https://||' -e 's|http://||' -e 's|/.*||')


# Create a directory for downloads
mkdir -p "../data/webpages/$DOMAIN"
cd "../data/webpages/$DOMAIN"

# Use wget with specific options for recursive download
wget \
    --recursive \
    --no-parent \
    --reject=css,js,jpg,jpeg,png,gif \
    --accept html \
    --domains="$DOMAIN" \
    --convert-links \
    --restrict-file-names=windows \
    --no-clobber \
    "$1"

echo "Download complete. Files saved in /data/webpages/$DOMAIN directory."

#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

DIRECTORY=$1

find "$DIRECTORY" -type f -name "*.zip" -execdir unzip {} \;
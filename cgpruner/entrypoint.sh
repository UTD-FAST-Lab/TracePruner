#!/bin/bash
# Ensure that the DATA_FOLDER environment variable is set (you can provide it via the command line when running Docker)
if [ -z "$DATA_FOLDER" ]; then
    echo "DATA_FOLDER not set. Exiting."
    exit 1
fi

# Create the data folder if it doesn't exist
mkdir -p "$DATA_FOLDER"

# Check if the NJR-1 dataset exists in the data folder
if [ ! -d "$DATA_FOLDER/njr-1_dataset" ]; then
    echo "Downloading NJR-1 dataset..."
    wget -q https://zenodo.org/records/4839913/files/njr-1_dataset.zip -O "$DATA_FOLDER/njr-1_dataset.zip"
    unzip -q "$DATA_FOLDER/njr-1_dataset.zip" -d "$DATA_FOLDER/njr-1_dataset"
    rm -f "$DATA_FOLDER/njr-1_dataset.zip"
else
    echo "NJR-1 DDataset already exists in $DATA_FOLDER"
fi

# Ensure the output folder structure exists
mkdir -p "$DATA_FOLDER/output/static_cgs"
mkdir -p "$DATA_FOLDER/output/stats"

echo "going to the shell"

# Start the container's main process (bash in this case)
# If no command is provided, default to bash
# if [ $# -eq 0 ]; then
#     exec /bin/bash
# else
#     exec "$@"
# fi

#!/bin/bash
rm -rf ~/.m2/repository/com/ibm/wala
echo "Building WALA..."

# Run the build process
./gradlew clean jar

echo "Transferring JAR files to script folders..."

# Define the source and destination directories
BASE_DIR=$(pwd) 
DEST_DIR="$HOME/projects/TracePruner/scripts/trace-generation/wala-jars/repository/com/ibm/wala"
DRIVER_DIR="$HOME/projects/TracePruner/scripts/trace-generation/driver/wala-project"

# Ensure destination directories exist
# mkdir -p "$DEST_DIR/core" "$DEST_DIR/util" "$DEST_DIR/shrike"

# delete the previous files from repo
rm -rf "$DEST_DIR/com.ibm.wala.core/1.5.9/com.ibm.wala.core-1.5.9.jar"
rm -rf "$DEST_DIR/com.ibm.wala.util/1.5.9/com.ibm.wala.util-1.5.9.jar"
rm -rf "$DEST_DIR/com.ibm.wala.shrike/1.5.9/com.ibm.wala.shrike-1.5.9.jar"


# Copy JAR files from the build directories to the corresponding folders
cp "$BASE_DIR/com.ibm.wala.core/build/libs/com.ibm.wala.core-1.5.9.jar" "$DEST_DIR/com.ibm.wala.core/1.5.9"
cp "$BASE_DIR/com.ibm.wala.util/build/libs/com.ibm.wala.util-1.5.9.jar" "$DEST_DIR/com.ibm.wala.util/1.5.9"
cp "$BASE_DIR/com.ibm.wala.shrike/build/libs/com.ibm.wala.shrike-1.5.9.jar" "$DEST_DIR/com.ibm.wala.shrike/1.5.9"

echo "JAR files transferred successfully."


# Rebuild the driver Maven project
echo "Rebuilding the driver Maven project..."
cd "$DRIVER_DIR" || { echo "Driver directory not found!"; exit 1; }
mvn clean install

echo "Driver Maven project built successfully."

rm -rf ~/.m2/repository/com/ibm/wala
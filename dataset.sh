#!/bin/bash

# This shell script will download the dataset from source

DIR="./Data/prototype/"

if [ -d "$DIR" ]; then
    echo "Prototype datset dir already exists"
else
    mkdir -p "$DIR"
    echo "Prototype dataset has been created"
fi

curl -q https://tomografia.es/data/train_SERAM.npy --output ./Data/prototype/train_SERAM.npy
curl -q https://tomografia.es/data/test1_SERAM.npy --output ./Data/prototype/test1_SERAM.npy
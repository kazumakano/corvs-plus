#!/bin/sh

mkdir --parents models/

echo Downloading model
curl --output models/model.safetensors https://zenodo.org/records/17745683/files/model.safetensors

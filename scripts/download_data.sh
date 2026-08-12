#!/bin/sh

mkdir --parents data/

echo Downloading README
curl --output data/README.md https://zenodo.org/records/17745683/files/README.md

echo Downloading and extracting trajectory data
curl --output data/trajectory.zip https://zenodo.org/records/17745683/files/trajectory.zip
unzip data/trajectory.zip -d data/trajectory/
rm data/trajectory.zip

echo Downloading and extracting sensor measurement data
curl --output data/sensor.zip https://zenodo.org/records/17745683/files/sensor.zip
unzip data/sensor.zip -d data/sensor/
rm data/sensor.zip

echo Downloading and extracting videos
curl --output data/video.zip https://zenodo.org/records/17745683/files/video.zip
unzip data/video.zip -d data/video/
rm data/video.zip

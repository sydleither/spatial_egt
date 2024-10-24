# Spatial Signatures of Games

## Overview
Coming soon

## Installation

### Repository
`git clone --recursive git@github.com:sydleither/spatial-egt.git`

### Java
Java version: 21.0.2

`cd in_silico_model`

`mkdir lib`

Install the jar files for [jackson-core-2.16.1](https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-core/2.16.1), [jackson-databind-2.16.1](https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-databind), and [jackson-annotations-2.16.1](https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-annotations/2.16.1)

Place the three jar files in lib/

`bash build.sh`

### Python
`conda create --name spatial-egt --file requirements.txt` or use a different virtual environment

### Test Installation
`cd in_silico_model`

`python3 generate_configs.py output test`
- Create directories and config for test data

`java -cp build/:lib/* SpatialEGT.SpatialEGT output test test 2D 0`
- Run model locally
- Output will be in output/test/test/0. The output, 2Dcoords.csv, records the coordinates of each cell at the end of the run.

`bash output/test/run.sh`
- Run model as a job on a HPCC
- This will require manual modification of `run_config.sb`

`bash output/test/visualize.sh`
- Visualize model locally
- Will create a pop-up visualization of the model to watch as it runs. Output will be in output/test/test/0. The output consists of images of the model at the start and end of the run, and a gif of the run.

## Replicate Results

### Generate *in silico* Data
`cd in_silico_model`

`python3 generate_configs.py ../data/in_silico raw`

`bash ../data/in_silico/raw/run`i`.sh`
- MSU HPCC can only accept 1000 jobs at a time so the submission scripts are batched.
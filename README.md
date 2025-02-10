# Spatial Signatures of Games

## Overview
Coming soon

## Installation

### Repository
`git clone --recursive git@github.com:sydleither/spatial-egt.git`

### Java
Java version: 21.0.2

`cd ABM`

`mkdir lib`

Install the jar files for [jackson-core-2.16.1](https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-core/2.16.1), [jackson-databind-2.16.1](https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-databind/2.16.1), and [jackson-annotations-2.16.1](https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-annotations/2.16.1)

Place the three jar files in `lib/`.

`bash build.sh`

### Python
`conda create --name spatial-egt --file requirements.txt` or use a different virtual environment.

### Test Installation
`cd ABM`

`python3 generate_configs.py output test 2D 1 42`
- Create directories and config for test data.
- The "output" and "test" arguments specify to save the data into a directory called "output/test", and "test" is the name of experiment to run. 2D specifies the dimension of data, 1 is the number of samples/models, and 42 is the seed.
- Along with creating the directories to save the data, this script also creates run.sh and visualize.sh files for running multiple models in one command. In this test case, only one model config is specified.

`java -cp build/:lib/* SpatialEGT.SpatialEGT output test test 2D 0`
- Run model locally.
- Output will be in output/test/test/0. The output, 2Dcoords.csv, records the coordinates of each cell at the end of the run.

`bash output/test/run.sh`
- Run model as a job on MSU's HPCC (optional).
- This will require manual modification of `run_config.sb` to remove my email and my lab's node.

`bash output/test/visualize.sh`
- Visualize model locally.
- Will create a pop-up visualization of the model to watch as it runs. Output will be in output/test/test/0. The output consists of images of the model at the start and end of the run, and a gif of the run.

## Replicate Results

### Generate *in silico* Data
`cd ABM`

`python3 generate_configs.py ../data/in_silico/raw HAL 2D 2500 42`
- Create directories, configs, and run scripts for the "raw" *in silico* data.
- If you want to generate the data locally, replace `sbatch run_config.sb` with `java -cp build/:lib/* SpatialEGT.SpatialEGT` in the generate_scripts function in generate_configs.py.

`bash ../data/in_silico/raw/HAL/run`{i}`.sh`
- Run the models.
- MSU HPCC can only accept 1000 jobs at a time so the submission scripts are batched ({i}).
- Data is saved in the form "data/in_silico/raw/HAL/{sample_id}/0/2Dcoords.csv".

`python3 -m data_processing.in_silico.payoff_raw_to_processed`
- Compile configs into a csv with the payoff values and game of each sample/model.
- Output saved at data/in_silico/processed/payoff.csv.

`python3 -m data_processing.in_silico.spatial_raw_to_processed`
- Clean up coords of cells at final timestep and save in one spot.
- Output saved at "data/in_silico/processed/HAL {sample_id}.csv"
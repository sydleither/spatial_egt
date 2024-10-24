# spatial-egt

## Overview

Coming soon
## Installation

### Repository
`git clone --recursive git@github.com:sydleither/spatial-egt.git`
### Java
Version: 21.0.2
- Install the jar files for [jackson-core-2.16.1](https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-core/2.16.1), [jackson-databind-2.16.1](https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-databind), and [jackson-annotations-2.16.1](https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-annotations/2.16.1)
- Create the directory in_silico_model/lib/ (`mkdir in_silico_model/lib`)
- Place the three jar files in lib/
- Run build.sh from within in_silico_model (`cd in_silico_model ; bash build.sh`)
### Python
`conda create --name spatial-egt --file requirements.txt`
or use a different virtual environment
## Test Installation
From within in_silico_model/
`python3 generate_configs.py output test`: create directories and config for test data
`java -cp build/:lib/* SpatialEGT.SpatialEGT output test test 2D 0`: run model locally
`bash output/test/run.sh`: run model as a job on a HPCC
- this will require manual modification of `run_config.sb`
`bash output/test/visualize.sh`: visualize model locally
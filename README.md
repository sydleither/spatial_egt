# Library for Game-Theoretic Spatial Analysis of Cells

## Overview
This repository offers supporting Python files for a project conducting spatial analysis of cells (agent-based, *in vitro*, or *in vivo*). This repo is meant to be used as a submodule for another project. Many of the spatial statistics are calculated using the [MuSpAn](https://www.muspan.co.uk/) package.

Examples of repos which implement spatial_egt successfully:

- [agent-based-games](https://github.com/sydleither/agent-based-games)

## Installation
By itself: `git clone --recursive git@github.com:sydleither/spatial_egt.git`

As a submodule:
- `git submodule add https://github.com/sydleither/spatial_egt`
- `git submodule update --init --recursive`

Conda enviornment: `conda create --name spatial_egt --file requirements.txt`

## Guide
Please note the following when using this library:

- A Python script called "spatial_database.py" must be in the parent directory. This defines which spatial statistics will be calculated on the data, and their parameters.

- Spatial data must be saved in data/{data_name}/processed. Processed data is named with the convention "{source} {sample_id}.csv". The csv should have the columns x,y,(z optionally),type.

- To calculate and label spatial data with their games, save "payoff.csv" into data/{data_name}/processed. This file should contain the columns source, sample_id, a, b, c, d (the payoff matrix parameters).

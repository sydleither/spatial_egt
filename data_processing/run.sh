#!/bin/sh
python3 -m data_processing.${1}.payoff_raw_to_processed
python3 -m data_processing.${1}.spatial_raw_to_processed
python3 -m data_processing.processed_to_features ${1}
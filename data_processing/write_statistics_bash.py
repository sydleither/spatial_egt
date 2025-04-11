"""Generate bash script for processing coordinates into spatial statistics.

Expected usage:
python3 -m spatial_egt.data_processing.write_statistics_bash
    data_type run_cmd file_name python_file (statistic_names)

Where:
data_type: the parent data dir
run_cmd: the command to use for running the spatial statistics, eg python3 or job.sb
file_name: name of resulting bash script
python_file: which file to run, eg processed_to_statistic
statistic_name: optional
    if provided the bash script will calculate the spatial statistic separately for each sample
    otherwise the bash script will run each spatial statistic over all samples
"""

import os
import sys

from feature_database import FEATURE_REGISTRY
from spatial_egt.common import get_data_path


def write_individual(run_cmd, python_file, data_type, statistic, file_name):
    """Write run for each sample of a given spatial statistic"""
    processed_path = get_data_path(data_type, "processed")
    sample_files = [x for x in os.listdir(processed_path) if x != "payoff.csv"]
    output = []
    for sample_file in sample_files:
        source = sample_file.split(" ")[0]
        sample = sample_file.split(" ")[1][:-4]
        output.append(f"{run_cmd} {python_file} {data_type} {statistic} {source} {sample}\n")
    output_batches = [output[i : i + 900] for i in range(0, len(output), 900)]
    for i, batch in enumerate(output_batches):
        with open(f"{file_name}_{data_type}_{statistic}_{i}.sh", "w", encoding="UTF-8") as f:
            for output_line in batch:
                f.write(output_line)
    os.mkdir(f"data/{data_type}/features/{statistic}")


def write_aggregated(run_cmd, python_file, data_type, statistic_names, file_name):
    """Write run for each spatial statistic"""
    output = []
    for statistic_name in statistic_names:
        output.append(f"{run_cmd} {python_file} {data_type} {statistic_name}\n")
    with open(f"{file_name}_{data_type}.sh", "w", encoding="UTF-8") as f:
        for output_line in output:
            f.write(output_line)


def main(data_type, run_cmd, file_name, python_file, statistic_name=None):
    """Generate and save bash script"""
    if statistic_name is None:
        write_aggregated(run_cmd, python_file, data_type, FEATURE_REGISTRY.keys(), file_name)
    else:
        write_individual(run_cmd, python_file, data_type, statistic_name, file_name)


if __name__ == "__main__":
    if len(sys.argv) in (5, 6):
        main(*sys.argv[1:])
    else:
        print("Please see the module docstring for usage instructions.")

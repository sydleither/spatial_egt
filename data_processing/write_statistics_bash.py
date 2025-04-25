"""Generate bash script for processing coordinates into spatial statistics.

Expected usage:
python3 -m spatial_egt.data_processing.write_statistics_bash
    data_type run_cmd file_name python_file (statistic_names)

Where:
data_type: the name of the directory in data/
run_cmd: the command to use for running the spatial statistics, eg "python3 -m" or "sbatch job.sb"
python_file: which file to run, eg spatial_egt.data_processing.processed_to_statistic
statistic_name: optional
    if provided the bash script will calculate the spatial statistic separately for each sample
    otherwise the bash script will run each spatial statistic over all samples
"""

import os
import sys

from spatial_database import STATISTIC_REGISTRY
from spatial_egt.common import get_data_path


def write_individual(run_cmd, python_file, data_type, statistic):
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
        with open(f"run_{data_type}_{statistic}_{i}.sh", "w", encoding="UTF-8") as f:
            for output_line in batch:
                f.write(output_line)
    os.mkdir(f"data/{data_type}/statistics/{statistic}")


def write_aggregated(run_cmd, python_file, data_type, statistic_names):
    """Write run for each spatial statistic"""
    output = []
    for statistic_name in statistic_names:
        output.append(f"{run_cmd} {python_file} {data_type} {statistic_name}\n")
    with open(f"run_{data_type}.sh", "w", encoding="UTF-8") as f:
        for output_line in output:
            f.write(output_line)


def main(data_type, run_cmd, python_file, statistic_name=None):
    """Generate and save bash script"""
    if statistic_name is None:
        write_aggregated(run_cmd, python_file, data_type, STATISTIC_REGISTRY.keys())
    else:
        write_individual(run_cmd, python_file, data_type, statistic_name)


if __name__ == "__main__":
    if len(sys.argv) in (4, 5):
        main(*sys.argv[1:])
    else:
        print("Please see the module docstring for usage instructions.")

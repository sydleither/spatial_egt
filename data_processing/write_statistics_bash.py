"""Generate bash script for processing coordinates into spatial statistics.

Expected usage:
python3 -m spatial_egt.data_processing.write_statistics_bash
    data_type time run_cmd file_name (statistic_name)

Where:
data_type: the name of the directory in data/
time: timepoint
run_cmd: the command to use for running the spatial statistics, eg "python3 -m" or "sbatch job.sb"
statistic_name: optional
    if provided the bash script will calculate the spatial statistic separately for each sample
    otherwise the bash script will run each spatial statistic over all samples
"""

import os
import sys

from spatial_database import STATISTIC_REGISTRY
from spatial_egt.common import get_data_path


def write_individual(run_cmd, python_file, data_type, time, statistic):
    """Write run for each sample of a given spatial statistic"""
    processed_path = get_data_path(data_type, "processed", time)
    output = []
    for sample_file in processed_path:
        source = sample_file.split(" ")[0]
        sample = sample_file.split(" ")[1][:-4]
        output.append(f"{run_cmd} {python_file} {data_type} {statistic} {time} {source} {sample}\n")
    output_batches = [output[i : i + 900] for i in range(0, len(output), 900)]
    for i, batch in enumerate(output_batches):
        with open(f"run_{data_type}_{time}_{statistic}_{i}.sh", "w", encoding="UTF-8") as f:
            for output_line in batch:
                f.write(output_line)
    get_data_path(data_type, f"statistics/{statistic}", time)


def write_aggregated(run_cmd, python_file, data_type, time, statistic_names):
    """Write run for each spatial statistic"""
    output = []
    for statistic_name in statistic_names:
        output.append(f"{run_cmd} {python_file} {data_type} {statistic_name} {time}\n")
    with open(f"run_{data_type}_{time}.sh", "w", encoding="UTF-8") as f:
        for output_line in output:
            f.write(output_line)
    get_data_path(data_type, "statistics", time)


def main(data_type, time, run_cmd, statistic_name=None):
    """Generate and save bash script"""
    python_file = "spatial_egt.data_processing.processed_to_statistic"
    if statistic_name is None:
        write_aggregated(run_cmd, python_file, data_type, time, STATISTIC_REGISTRY.keys())
    else:
        write_individual(run_cmd, python_file, data_type, time, statistic_name)


if __name__ == "__main__":
    if len(sys.argv) in (4, 5):
        main(*sys.argv[1:])
    else:
        print("Please see the module docstring for usage instructions.")

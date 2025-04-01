import os
import sys

from common import get_data_path
from data_processing.feature_database import FEATURE_REGISTRY


def write_individual(run_cmd, python_file, data_type, feature_names, file_preamble):
    processed_path = get_data_path(data_type, "processed")
    sample_files = os.listdir(processed_path)
    sample_files = [x for x in sample_files if x != "payoff.csv"]
    for feature_name in feature_names:
        output = []
        for sample_file in sample_files:
            source = sample_file.split(" ")[0]
            sample = sample_file.split(" ")[1][:-4]
            output.append(f"{run_cmd} {python_file} {data_type} {feature_name} {source} {sample}\n")
        output_batches = [output[i:i + 900] for i in range(0, len(output), 900)]
        for i,batch in enumerate(output_batches):
            with open(f"{file_preamble}_{data_type}_{feature_name}_{i}.sh", "w") as f:
                for output_line in batch:
                    f.write(output_line)
        os.mkdir(f"data/{data_type}/features/{feature_name}")


def write_aggregated(run_cmd, python_file, data_type, feature_names, file_preamble):
    output = []
    for feature_name in feature_names:
        output.append(f"{run_cmd} {python_file} {data_type} {feature_name}\n")
    with open(f"{file_preamble}_{data_type}.sh", "w") as f:
        for output_line in output:
            f.write(output_line)


def main(data_type, run_loc, file_preamble, python_file, feature_names=None):
    if run_loc == "hpcc":
        run_cmd = "sbatch job.sb"
    elif run_loc == "local":
        run_cmd = "python3 -m"
    else:
        print(f"Invalid run location given: {run_loc}")
        return

    if feature_names is None:
        feature_names = FEATURE_REGISTRY.keys()
        write_aggregated(run_cmd, python_file, data_type, feature_names, file_preamble)
    else:
        write_individual(run_cmd, python_file, data_type, feature_names, file_preamble)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) > 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        print("Please provide the data type,")
        print("'local' or 'hpcc' for where the features will be processed,")
        print("the script name, the python module (e.g. data_processing.processed_to_feature),")
        print("and (optionally) the feature names if running samples independently.")

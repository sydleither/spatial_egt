import os
import sys

from config_utils import lhs, write_config, write_run_scripts


def main(data_dir, experiment_name, num_samples, seed):
    replicates = 1
    run_command = "sbatch run_config.sb"
    space = "2D"

    run_output = []
    samples = lhs(num_samples,
                  ["A", "B", "C", "D", "fr", "cells"],
                  [0, 0, 0, 0, 0.25, 625],
                  [0.1, 0.1, 0.1, 0.1, 0.75, 15000],
                  [False, False, False, False, False, True],
                  seed=seed)

    for s,sample in enumerate(samples):
        config_name = str(s)
        for r in range(replicates):
            os.makedirs(data_dir+"/"+experiment_name+"/"+config_name+"/"+str(r))
        payoff = [sample["A"], sample["B"], sample["C"], sample["D"]]
        write_config(data_dir, experiment_name, config_name,
                     payoff, sample["cells"], sample["fr"])
        run_output.append(f"{run_command} {data_dir} {experiment_name} {config_name} {space}\n")

    write_run_scripts(data_dir, experiment_name, run_output)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Please provide the data dir, experiment name, number of samples, and the seed.")
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

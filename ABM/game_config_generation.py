import os
import sys

from config_utils import write_config, write_run_scripts


def main(data_dir, experiment_name):
    replicates = 1
    run_command = "java -cp build/:lib/* SpatialEGT.SpatialEGT"
    space = "2D"
    end_time = 250

    high = 0.06
    low = 0.01
    samples = {
        "Sensitive_Wins":[high, high, low, low],
        "Coexistence":[low, high, high, low],
        "Bistability":[high, low, low, high],
        "Resistant_Wins":[low, low, high, high]
    }

    run_output = []
    run_str = f"{run_command} {data_dir} {experiment_name}"
    for game,payoff in samples.items():
        config_name = game
        for r in range(replicates):
            os.makedirs(data_dir+"/"+experiment_name+"/"+config_name+"/"+str(r))
            run_output.append(f"{run_str} {config_name} {space} {r} {end_time}\n")
            run_output.append(f"{run_str} {config_name} {space} {r}\n")
        write_config(data_dir, experiment_name, config_name, payoff,
                     15625, 0.5, write_freq=end_time, ticks=end_time)
    write_run_scripts(data_dir, experiment_name, run_output)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide the data dir and experiment name.")
    else:
        main(sys.argv[1], sys.argv[2])

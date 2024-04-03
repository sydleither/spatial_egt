import json
import os
import sys


def experiment_config(exp_dir, config_name, deathRate, drugGrowthReduction, numCells, proportionResistant, numDays, payoff):
    config = {
        "x": 125,
        "y": 125,
        "deathRate": deathRate,
        "drugGrowthReduction": drugGrowthReduction,
        "numCells": numCells,
        "proportionResistant": proportionResistant,
        "numDays": numDays,
        "egt": True,
        "A": payoff[0],
        "B": payoff[1],
        "C": payoff[2],
        "D": payoff[3]
    }

    config_path = f"output/{exp_dir}/{config_name}/{config_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def initial_games(exp_dir):
    if not os.path.exists("output/"+exp_dir):
        os.makedirs("output/"+exp_dir)
    config_names = []

    gw = 0.03
    gm = 0.8*gw

    names = ["competition", "no_game", "coexistance", "bistability"]
    bmws = [-0.007, 0, 0.007, -0.007]
    bwms = [0.0, 0.0, 0.0, -0.02]
    for i in range(len(names)):
        bmw = bmws[i]
        bwm = bwms[i]
        payoff = [gw, round(gw+bwm, 3), round(gm+bmw, 3), gm]

        exp_name = names[i]
        os.makedirs("output/"+exp_dir+"/"+exp_name)
        experiment_config(exp_dir, exp_name, 0.0081, 0.5, 8125, 0.001, 20000, payoff)
        config_names.append(exp_name)

    submit_output, analysis_output = generate_scripts_batch(exp_dir, config_names)
    return submit_output, analysis_output


def generate_scripts_batch(exp_dir, config_names):
    run_command = "/usr/bin/env /opt/software/Java/1.8.0_152/bin/java -cp /tmp/cp_ds16usm4x03dkwda919iyg6pd.jar SpatialEGT.SpatialEGT"

    submit_output = []
    analysis_output = []
    for config_name in config_names:
        for space in ["WM", "2D", "3D"]:
            submit_output.append(f"{run_command} {exp_dir} {config_name} {space}\n")
            analysis_output.append(f"python3 analysis/pop_over_time.py {exp_dir} {config_name} {space}\n")

    return submit_output, analysis_output


def write_scripts_batch(exp_dir, submit_output, analysis_output):
    with open(f"output/run_{exp_dir}_experiments", "w") as f:
        for output_line in submit_output:
            f.write(output_line)

    with open(f"output/analyze_{exp_dir}_experiments", "w") as f:
        for output_line in analysis_output:
            f.write(output_line)


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    submit_output = []
    analysis_output = []
    if experiment_name == "games":
        s, a = initial_games(experiment_name)
        submit_output += s
        analysis_output += a
    else:
        print("Invalid experiment name.")
        exit()
    write_scripts_batch(experiment_name, submit_output, analysis_output)
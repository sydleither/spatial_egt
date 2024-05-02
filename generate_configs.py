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


def initial_games(exp_dir, names, bmws, bwms, turnover=0.009, drug_reduction= 0.5, prop_res=0.01, runtime=50000):
    if not os.path.exists("output/"+exp_dir):
        os.mkdir("output/"+exp_dir)
    config_names = []

    gw = 0.03
    gm = 0.8*gw

    for i in range(len(names)):
        bmw = bmws[i]
        bwm = bwms[i]
        payoff = [gw, round(gw+bwm, 3), round(gm+bmw, 3), gm]

        exp_name = names[i]
        os.mkdir("output/"+exp_dir+"/"+exp_name)
        for i in range(10):
            os.mkdir("output/"+exp_dir+"/"+exp_name+"/"+str(i))
        experiment_config(exp_dir, exp_name, turnover, drug_reduction, 4375, prop_res, runtime, payoff)
        config_names.append(exp_name)

    submit_output, analysis_output = generate_scripts_batch(exp_dir, config_names)
    return submit_output, analysis_output


def generate_scripts_batch(exp_dir, config_names):
    submit_output = []
    analysis_output = []
    for config_name in config_names:
        for space in ["WM", "2D", "3D"]:
            if space == "WM":
                config_type = "long"
            else:
                config_type = "short"
            submit_output.append(f"sbatch run_config_{config_type}.sb {exp_dir} {config_name} {space}\n")
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
        names = ["competition", "no_game", "coexistance"]
        bmws = [-0.007, 0, 0.007]
        bwms = [0.0, 0.0, 0.0]
        s, a = initial_games(experiment_name, names, bmws, bwms)
        submit_output += s
        analysis_output += a
    elif experiment_name == "gamesd":
        names = ["competition", "no_game", "coexistance"]
        bmws = [-0.007, 0, 0.007]
        bwms = [0.0, 0.0, 0.0]
        s, a = initial_games(experiment_name, names, bmws, bwms, turnover=0.015)
        submit_output += s
        analysis_output += a
    elif experiment_name == "coexist":
        names = ["14", "25", "33", "40", "45", "50"]
        bmws = [0.007, 0.008, 0.009, 0.010, 0.011, 0.012]
        bwms = [0.0]*len(bmws)
        s, a = initial_games(experiment_name, names, bmws, bwms)
        submit_output += s
        analysis_output += a
    elif experiment_name == "test":
        names = ["coexistance"]
        bmws = [0.007]
        bwms = [0.0]
        s, a = initial_games(experiment_name, names, bmws, bwms, turnover=0.015, runtime=100)
        submit_output += s
        analysis_output += a
    elif experiment_name == "bistability":
        names = ["bistability", "no_game"]
        bmws = [-0.007, 0]
        bwms = [-0.02, 0]
        for prop_res in [0.3, 0.5, 0.7]:
            s, a = initial_games(experiment_name, [x+str(prop_res)[-1] for x in names], bmws, bwms, prop_res=prop_res)
            submit_output += s
            analysis_output += a
    elif experiment_name == "drug":
        names = ["no_game"]
        bmws = [0]
        bwms = [0]
        for drug_reduction in [0.5, 0.6, 0.7, 0.8]:
            s, a = initial_games(experiment_name, [x+str(drug_reduction)[-1] for x in names], bmws, bwms, drug_reduction=drug_reduction)
            submit_output += s
            analysis_output += a
    else:
        print("Invalid experiment name.")
        exit()
    write_scripts_batch(experiment_name, submit_output, analysis_output)
    print("Make sure you recompile SpatialEGT before pushing jobs:")
    print("javac -d \"build\" -cp \"lib/*\" @sources.txt")
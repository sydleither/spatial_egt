import json
import os
import sys


def experiment_config(exp_dir, config_name, runNull, runAdaptive, runContinuous, writePopFrequency, writeFsFrequency,
                      writePcFrequency, numTicks, radius, deathRate, drugGrowthReduction, numCells, 
                      proportionResistant, adaptiveTreatmentThreshold, initialTumor, payoff):
    config = {
        "null": runNull,
        "adaptive": runAdaptive,
        "continuous": runContinuous,
        "visualizationFrequency": 0,
        "writePopFrequency": writePopFrequency,
        "writePcFrequency": writePcFrequency,
        "writeFsFrequency": writeFsFrequency,
        "x": 125,
        "y": 125,
        "neighborhoodRadius": radius,
        "numTicks": numTicks,
        "deathRate": deathRate,
        "drugGrowthReduction": drugGrowthReduction,
        "numCells": numCells,
        "proportionResistant": proportionResistant,
        "adaptiveTreatmentThreshold": adaptiveTreatmentThreshold,
        "initialTumor": initialTumor,
        "A": payoff[0],
        "B": payoff[1],
        "C": payoff[2],
        "D": payoff[3]
    }

    config_path = f"output/{exp_dir}/{config_name}/{config_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def initial_games(exp_dir, names, bmws, bwms, gw=0.03, runNull=1, runAdaptive=0, runContinuous=1, writePopFrequency=1, writeFsFrequency=0,
                  writePcFrequency=500, radius=1, turnover=0.009, drug_reduction=0.5, init_cells=4375, prop_res=0.01, 
                  adaptiveTreatmentThreshold=0.5, initialTumor=0, runtime=50000):
    if not os.path.exists("output/"+exp_dir):
        os.mkdir("output/"+exp_dir)
    config_names = []

    gm = 0.8*gw

    for i in range(len(names)):
        bmw = bmws[i]
        bwm = bwms[i]
        payoff = [gw, round(gw+bwm, 3), round(gm+bmw, 3), gm]

        exp_name = names[i]
        os.mkdir("output/"+exp_dir+"/"+exp_name)
        for i in range(10):
            os.mkdir("output/"+exp_dir+"/"+exp_name+"/"+str(i))
        experiment_config(exp_dir, exp_name, runNull, runAdaptive, runContinuous, writePopFrequency, writeFsFrequency,
                          writePcFrequency, runtime, radius, turnover, drug_reduction, init_cells, 
                          prop_res, adaptiveTreatmentThreshold, initialTumor, payoff)
        config_names.append(exp_name)

    submit_output, analysis_output = generate_scripts_batch(exp_dir, config_names)
    return submit_output, analysis_output


def custom_games(exp_dir, names, a, b, c, d, runNull=1, runAdaptive=0, runContinuous=1, writePopFrequency=1, writeFsFrequency=0,
                  writePcFrequency=500, radius=1, turnover=0.009, drug_reduction=0.5, init_cells=4375, prop_res=0.01, 
                  adaptiveTreatmentThreshold=0.5, initialTumor=0, runtime=500):
    if not os.path.exists("output/"+exp_dir):
        os.mkdir("output/"+exp_dir)
    config_names = []

    for i in range(len(names)):
        payoff = [a[i], b[i], c[i], d[i]]

        exp_name = names[i]
        os.mkdir("output/"+exp_dir+"/"+exp_name)
        for i in range(10):
            os.mkdir("output/"+exp_dir+"/"+exp_name+"/"+str(i))
        experiment_config(exp_dir, exp_name, runNull, runAdaptive, runContinuous, writePopFrequency, writeFsFrequency,
                          writePcFrequency, runtime, radius, turnover, drug_reduction, init_cells, 
                          prop_res, adaptiveTreatmentThreshold, initialTumor, payoff)
        config_names.append(exp_name)

    submit_output, analysis_output = generate_scripts_batch(exp_dir, config_names)
    return submit_output, analysis_output


def generate_scripts_batch(exp_dir, config_names):
    submit_output = []
    analysis_output = []
    for config_name in config_names:
        for space in ["2D"]:
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

    # with open(f"output/analyze_{exp_dir}_experiments", "w") as f:
    #     for output_line in analysis_output:
    #         f.write(output_line)


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    submit_output = []
    analysis_output = []
    if experiment_name == "games":
        names = ["competition", "no_game", "coexistance"]
        bmws = [-0.007, 0, 0.007]
        bwms = [0.0, 0.0, 0.0]
        s, a = initial_games(experiment_name, names, bmws, bwms, runtime=10000)
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
        names = ["14", "25", "33", "40", "45", "50", "60", "70", "80", "90"]
        bmws = [0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.015, 0.020, 0.030, 0.060]
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
        for drug_reduction in [0.5, 0.6, 0.7, 0.8, 0.9]:
            s, a = initial_games(experiment_name, [x+str(drug_reduction)[-1] for x in names], bmws, bwms, drug_reduction=drug_reduction, prop_res=0.001, runtime=10000)
            submit_output += s
            analysis_output += a
    elif experiment_name == "threshold":
        names = ["competition", "no_game", "coexistance"]
        bmws = [-0.007, 0, 0.007]
        bwms = [0.0, 0.0, 0.0]
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            s, a = initial_games(experiment_name, [x+str(threshold)[-1] for x in names],
                                 bmws, bwms, drug_reduction=0.9, prop_res=0.01, 
                                 adaptiveTreatmentThreshold=threshold, runtime=5000)
            submit_output += s
            analysis_output += a
    elif experiment_name == "threshold_d":
        names = ["competition", "no_game", "coexistance"]
        bmws = [-0.007, 0, 0.007]
        bwms = [0.0, 0.0, 0.0]
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            s, a = initial_games(experiment_name, [x+str(threshold)[-1] for x in names],
                                 bmws, bwms, drug_reduction=0.9, init_cells=11250, prop_res=0.01, 
                                 adaptiveTreatmentThreshold=threshold, runtime=5000)
            submit_output += s
            analysis_output += a
    elif experiment_name == "threshold_dfr":
        names = ["competition", "no_game", "coexistance"]
        bmws = [-0.007, 0, 0.007]
        bwms = [0.0, 0.0, 0.0]
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            s, a = initial_games(experiment_name, [x+str(threshold)[-1] for x in names],
                                 bmws, bwms, drug_reduction=0.9, init_cells=11250, prop_res=0.001, 
                                 adaptiveTreatmentThreshold=threshold, runtime=10000)
            submit_output += s
            analysis_output += a
    elif experiment_name == "gameshood":
        names = ["competition", "no_game", "coexistance"]
        bmws = [-0.007, 0, 0.007]
        bwms = [0.0, 0.0, 0.0]
        for radius in [1, 2, 3, 4, 5]:
            s, a = initial_games(experiment_name, [x+str(radius) for x in names], bmws, bwms, radius=radius)
            submit_output += s
            analysis_output += a
    elif experiment_name == "gamespc":
        names = ["competition", "no_game", "coexistance"]
        bmws = [-0.007, 0, 0.007]
        bwms = [0.0, 0.0, 0.0]
        for fr in ["0.050", "0.010", "0.002"]:
            s, a = initial_games(experiment_name, [x+fr[2:] for x in names], bmws, bwms, init_cells=2500, prop_res=float(fr), runtime=500, runContinuous=0)
            submit_output += s
            analysis_output += a
    elif experiment_name == "gamespc_sw":
        names = ["max", "half", "zero", "min"]
        bmws = [0.005, 0.00025, 0.0, -0.024]
        bwms = [0.0, 0.0, 0.0, 0.0]
        s, a = initial_games(experiment_name, names, bmws, bwms, init_cells=2500, prop_res=0.05, runtime=500, runContinuous=0, writePcFrequency=100)
        submit_output += s
        analysis_output += a
    elif experiment_name == "gamespc_co":
        names = ["25", "50", "75"]
        bmws = [0.008, 0.012, 0.024]
        bwms = [0.0, 0.0, 0.0]
        s, a = initial_games(experiment_name, names, bmws, bwms, init_cells=2500, prop_res=0.05, runtime=500, runContinuous=0, writePcFrequency=100)
        submit_output += s
        analysis_output += a
    elif experiment_name == "gamespc_bi":
        names = ["25", "50", "75"]
        bmws = [0.00133, -0.008, -0.036]
        bwms = [-0.02, -0.02, -0.02]
        s, a = initial_games(experiment_name, names, bmws, bwms, init_cells=2500, prop_res=0.05, runtime=500, runContinuous=0, writePcFrequency=100)
        submit_output += s
        analysis_output += a
    elif experiment_name == "gamespc_rw":
        names = ["min", "half", "max"]
        bmws = [0.004, 0.0455, 0.087]
        bwms = [-0.004, -0.004, -0.004]
        s, a = initial_games(experiment_name, names, bmws, bwms, gw=0.015, init_cells=2500, prop_res=0.05, runtime=500, runContinuous=0, writePcFrequency=100)
        submit_output += s
        analysis_output += a
    elif experiment_name == "gamespc_all":
        games = ["sensitive", "coexistence", "bistability", "resistant"]
        betas = [["max", "half", "zero", "min"], ["25", "50", "75"], ["25", "50", "75"], ["min", "half", "max"]]
        bmws = [[0.005, 0.00025, 0.0, -0.024], [0.008, 0.012, 0.024], [0.00133, -0.008, -0.036], [0.004, 0.0455, 0.087]]
        bwms = [[0.0]*4, [0.0]*3, [-0.02]*3, [-0.004]*3]
        gws = [0.03, 0.03, 0.03, 0.015]
        for i in range(len(games)):
            names = [games[i]+"_"+betas[i][x] for x in range(len(betas[i]))]
            s, a = initial_games(experiment_name, names, bmws[i], bwms[i], gw=gws[i], init_cells=100, 
                                 prop_res=0.05, runtime=500, runContinuous=0, writePcFrequency=100, radius=3)
            submit_output += s
            analysis_output += a
    elif experiment_name == "custom_gamespc_all":
        games = ["sensitive", "coexistence", "bistability", "resistant"]
        subgames = [["agtb", "bgta"], ["equal"], ["equal"], ["cgtd", "dgtc"]]
        pa = [[0.09, 0.06], [0.03], [0.06], [0.03, 0.0]]
        pb = [[0.06, 0.09], [0.06], [0.03], [0.0, 0.03]]
        pc = [[0.06, 0.03], [0.06], [0.03], [0.06, 0.03]]
        pd = [[0.03, 0.06], [0.03], [0.06], [0.03, 0.06]]
        for i in range(len(games)):
            names = [games[i]+"_"+subgames[i][x] for x in range(len(subgames[i]))]
            s, a = custom_games(experiment_name, names, a=pa[i], b=pb[i], c=pc[i], d=pd[i], initialTumor=2,
                                 init_cells=900, prop_res=0.5, runtime=500, runContinuous=0, writePcFrequency=50, radius=3)
            submit_output += s
            analysis_output += a
    elif experiment_name == "gamespc_within":
        games = ["sensitive", "coexistence"]
        subgames = [["agtb", "bgta", "equal"], ["bgtc", "cgtb", "equal"]]
        pa = [[0.09, 0.06, 0.06], [0.03, 0.06, 0.03]]
        pb = [[0.06, 0.09, 0.06], [0.09, 0.06, 0.06]]
        pc = [[0.06, 0.03, 0.03], [0.06, 0.09, 0.06]]
        pd = [[0.03, 0.06, 0.03], [0.06, 0.03, 0.03]]
        for i in range(len(games)):
            names = [games[i]+"_"+subgames[i][x] for x in range(len(subgames[i]))]
            s, a = custom_games(experiment_name, names, a=pa[i], b=pb[i], c=pc[i], d=pd[i], initialTumor=3, turnover=0.018,
                                 init_cells=15625, prop_res=0.5, runtime=2000, runContinuous=0, writePcFrequency=200, radius=3)
            submit_output += s
            analysis_output += a
    elif experiment_name == "gamespc_within2":
        games = ["bistability", "resistant"]
        subgames = [["equal", "agtd", "dgta"], ["cgtd", "dgtc", "equal"]]
        pa = [[0.06, 0.09, 0.06], [0.06, 0.03, 0.03]]
        pb = [[0.03, 0.03, 0.06], [0.03, 0.06, 0.03]]
        pc = [[0.03, 0.06, 0.03], [0.09, 0.06, 0.06]]
        pd = [[0.06, 0.06, 0.09], [0.06, 0.09, 0.06]]
        for i in range(len(games)):
            names = [games[i]+"_"+subgames[i][x] for x in range(len(subgames[i]))]
            s, a = custom_games(experiment_name, names, a=pa[i], b=pb[i], c=pc[i], d=pd[i], initialTumor=1, turnover=0.018,
                                 init_cells=15625, prop_res=0.5, runtime=2000, runContinuous=0, writePcFrequency=200, radius=3)
            submit_output += s
            analysis_output += a
    elif experiment_name == "at_paramsweep":
        names = ["competition", "no_game", "coexistence"]
        bmws = [-0.007, 0, 0.007]
        bwms = [0.0, 0.0, 0.0]
        for i,threshold in enumerate([0.3, 0.5, 0.7]):
            for j,fr in enumerate([0.01, 0.05, 0.1]):
                for k,cells in enumerate([1875, 6250, 11250]):
                    subexp_names = [f"{x}_thr{i}_fr{j}_c{k}" for x in names]
                    s, a = initial_games(experiment_name, subexp_names, bmws, bwms,
                                         radius=2, runAdaptive=1, writePcFrequency=0,
                                         drug_reduction=0.9, init_cells=cells, prop_res=fr, 
                                         adaptiveTreatmentThreshold=threshold, runtime=10000)
                    submit_output += s
                    analysis_output += a
    elif experiment_name == "at_gamesweep":
        names = ["competition", "no_game", "coexistence"]
        bmws = [-0.007, 0, 0.007]
        bwms = [0.0, 0.0, 0.0]
        s, a = initial_games(experiment_name, names, bmws, bwms, runNull=1,
                                radius=1, runAdaptive=1, writePcFrequency=0, writeFsFrequency=1,
                                drug_reduction=0.9, init_cells=11250, prop_res=0.01, 
                                adaptiveTreatmentThreshold=0.3, runtime=5000)
        submit_output += s
        analysis_output += a
    else:
        print("Invalid experiment name.")
        exit()
    write_scripts_batch(experiment_name, submit_output, analysis_output)
    print("Make sure you recompile SpatialEGT before pushing jobs:")
    print("javac -d \"build\" -cp \"lib/*\" @sources.txt")
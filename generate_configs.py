import json
import os
import sys

from scipy.stats import qmc


def experiment_config(exp_dir, config_name, runNull, runAdaptive, runContinuous, writePopFrequency, writePcFrequency,
                      writeFsFrequency, writeModelFrequency, numTicks, radius, deathRate, drugGrowthReduction,
                      numCells, proportionResistant, adaptiveTreatmentThreshold, initialTumor, toyGap, payoff):
    config = {
        "null": runNull,
        "adaptive": runAdaptive,
        "continuous": runContinuous,
        "writePopFrequency": writePopFrequency,
        "writePcFrequency": writePcFrequency,
        "writeFsFrequency": writeFsFrequency,
        "writeModelFrequency": writeModelFrequency,
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
        "toyGap": toyGap,
        "A": payoff[0],
        "B": payoff[1],
        "C": payoff[2],
        "D": payoff[3]
    }

    config_path = f"output/{exp_dir}/{config_name}/{config_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def sample_games(exp_dir, name, a, b, c, d, runNull=1, runAdaptive=0, runContinuous=0, 
                 writePopFrequency=1, writePcFrequency=0, writeFsFrequency=0, writeModelFrequency=0, 
                 radius=1, turnover=0.009, drug_reduction=0.5, init_cells=4375, prop_res=0.01, 
                 adaptiveTreatmentThreshold=0.5, initialTumor=0, toyGap=5, runtime=500, 
                 replicates=10, spaces=["2D", "3D", "WM"]):
    payoff = [a, b, c, d]
    exp_name = name
    os.makedirs("output/"+exp_dir+"/"+exp_name)
    for i in range(replicates):
        os.mkdir("output/"+exp_dir+"/"+exp_name+"/"+str(i))
    experiment_config(exp_dir, exp_name, runNull, runAdaptive, runContinuous, writePopFrequency, 
                      writePcFrequency, writeFsFrequency, writeModelFrequency, runtime, radius, 
                      turnover, drug_reduction, init_cells, prop_res, adaptiveTreatmentThreshold, 
                      initialTumor, toyGap, payoff)
    submit_output, analysis_output = generate_scripts_batch(exp_dir, [exp_name], spaces)
    return submit_output, analysis_output


def generate_scripts_batch(exp_dir, config_names, spaces):
    submit_output = []
    analysis_output = []
    for config_name in config_names:
        for space in spaces:
            if space == "WM":
                config_type = "long"
            else:
                config_type = "short"
            submit_output.append(f"sbatch run_config_{config_type}.sb {exp_dir} {config_name} {space}\n")
            analysis_output.append(f"java -cp build/:lib/* SpatialEGT.SpatialEGT {exp_dir} {config_name} {space} 0 2000\n")
    return submit_output, analysis_output


def write_scripts_batch(exp_dir, submit_output, analysis_output):
    with open(f"output/run_{exp_dir}_experiments", "w") as f:
        for output_line in submit_output:
            f.write(output_line)

    with open(f"output/analyze_{exp_dir}_experiments", "w") as f:
        for output_line in analysis_output:
            f.write(output_line)


def lhs_sample(num_samples, param_names, lower_bounds, upper_bounds, ints, seed=42):
    sampler = qmc.LatinHypercube(d=len(lower_bounds), seed=seed)
    unscaled_sample = sampler.random(n=num_samples)
    sample = qmc.scale(unscaled_sample, lower_bounds, upper_bounds).tolist()
    sampled_params = [{param_names[i]:round(s[i]) if ints[i] else round(s[i], 2) for i in range(len(s))} for s in sample]
    return sampled_params


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    submit_output = []
    analysis_output = []
    N = 2500
    if experiment_name == "sample":
        experiment_name = "sample"+str(N)
        samples = lhs_sample(N, 
                             ["A", "B", "C", "D", "fr", "cells"],
                             [0, 0, 0, 0, 0.1, 625], 
                             [0.1, 0.1, 0.1, 0.1, 0.9, 15625], 
                             [False, False, False, False, False, True], 
                             seed=42)
        for i in range(N):
            sample = samples[i]
            s, a = sample_games(writePopFrequency=250, writeModelFrequency=250, writeFsFrequency=250,
                                writePcFrequency=250,
                                runtime=250, exp_dir=experiment_name, name=str(i), a=sample["A"], 
                                b=sample["B"], c=sample["C"], d=sample["D"], initialTumor=0, 
                                turnover=0.009, init_cells=sample["cells"], prop_res=sample["fr"], 
                                runContinuous=0, radius=2, replicates=1, spaces=["2D"])
            submit_output += s
            analysis_output += a
    elif experiment_name == "games":
        games = ["sensitive", "coexistence", "bistability", "resistant"]
        subgames = [["agtb", "bgta", "equal"], ["bgtc", "cgtb", "equal"],
                    ["equal", "agtd", "dgta"], ["cgtd", "dgtc", "equal"]]
        pa = [[0.09, 0.06, 0.06], [0.03, 0.06, 0.03], [0.06, 0.09, 0.06], [0.06, 0.03, 0.03]]
        pb = [[0.06, 0.09, 0.06], [0.09, 0.06, 0.06], [0.03, 0.03, 0.06], [0.03, 0.06, 0.03]]
        pc = [[0.06, 0.03, 0.03], [0.06, 0.09, 0.06], [0.03, 0.06, 0.03], [0.09, 0.06, 0.06]]
        pd = [[0.03, 0.06, 0.03], [0.06, 0.03, 0.03], [0.06, 0.06, 0.09], [0.06, 0.09, 0.06]]
        for i in range(len(games)):
            for j in range(len(subgames[i])):
                name = games[i]+"_"+subgames[i][j]
                s, a = sample_games(writePopFrequency=500, writeModelFrequency=0, writeFsFrequency=500,
                                    writePcFrequency=0,
                                    runtime=500, exp_dir=experiment_name, name=name, a=pa[i][j], 
                                    b=pb[i][j], c=pc[i][j], d=pd[i][j], initialTumor=0, 
                                    turnover=0.009, init_cells=15625, prop_res=0.5, 
                                    runContinuous=0, radius=2, replicates=5, spaces=["2D"])
                submit_output += s
                analysis_output += a
    else:
        print("Invalid experiment name.")
        exit()
    write_scripts_batch(experiment_name, submit_output, analysis_output)
    print("Make sure you recompile SpatialEGT before pushing jobs:")
    print("javac -d \"build\" -cp \"lib/*\" @sources.txt")

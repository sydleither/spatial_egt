import json
import os
import sys

from scipy.stats import qmc


def experiment_config(data_dir, exp_dir, config_name, runNull, runAdaptive,
                      runContinuous, writeModelFrequency, numTicks, radius, 
                      deathRate, drugGrowthReduction, numCells, proportionResistant, 
                      adaptiveTreatmentThreshold, initialTumor, toyGap, payoff):
    config = {
        "null": runNull,
        "adaptive": runAdaptive,
        "continuous": runContinuous,
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

    config_path = f"{data_dir}/{exp_dir}/{config_name}/{config_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def sample_games(data_dir, exp_dir, name, a, b, c, d, runNull=1, runAdaptive=0, runContinuous=0, 
                 writeModelFrequency=500, radius=1, turnover=0.009, drug_reduction=0.5, 
                 init_cells=4375, prop_res=0.5, adaptiveTreatmentThreshold=0.5, 
                 initialTumor=0, toyGap=5, runtime=500, replicates=10, spaces=["2D","3D","WM"]):
    payoff = [a, b, c, d]
    for i in range(replicates):
        os.makedirs(data_dir+"/"+exp_dir+"/"+name+"/"+str(i))
    experiment_config(data_dir, exp_dir, name, runNull, runAdaptive, runContinuous, 
                      writeModelFrequency, runtime, radius, turnover, drug_reduction, 
                      init_cells, prop_res, adaptiveTreatmentThreshold, initialTumor, 
                      toyGap, payoff)
    run_output, visual_output = generate_scripts(data_dir, exp_dir, name, spaces, runtime)
    return run_output, visual_output


def generate_scripts(data_dir, exp_dir, config_name, spaces, viz_freq):
    run_output = []
    visual_output = []
    run_command = "java -cp build/:lib/* SpatialEGT.SpatialEGT"
    for space in spaces:
        run_output.append(f"sbatch run_config.sb {data_dir} {exp_dir} {config_name} {space}\n")
        visual_output.append(f"{run_command} {data_dir} {exp_dir} {config_name} {space} 0 {viz_freq}\n")
    return run_output, visual_output


def write_scripts(data_dir, exp_dir, run_output, visual_output, i=""):
    with open(f"{data_dir}/{exp_dir}/run{i}.sh", "w") as f:
        for output_line in run_output:
            f.write(output_line)
    with open(f"{data_dir}/{exp_dir}/visualize{i}.sh", "w") as f:
        for output_line in visual_output:
            f.write(output_line)


def lhs_sample(num_samples, param_names, lower_bounds, upper_bounds, ints, seed=42):
    sampler = qmc.LatinHypercube(d=len(lower_bounds), seed=seed)
    unscaled_sample = sampler.random(n=num_samples)
    sample = qmc.scale(unscaled_sample, lower_bounds, upper_bounds).tolist()
    sampled_params = [{param_names[i]:round(s[i]) if ints[i] else round(s[i], 2)
                       for i in range(len(s))} for s in sample]
    return sampled_params


def main(data_dir, experiment_name):
    run_output = []
    visual_output = []
    if experiment_name == "test":
        s, v = sample_games(data_dir=data_dir, exp_dir=experiment_name, runtime=100,
                            a=0.03, b=0.03, c=0.02, d=0.02, radius=2, replicates=1,
                            spaces=["2D"], name="test", writeModelFrequency=100)
        run_output += s
        visual_output += v
    elif experiment_name == "HAL2D":
        N = 2500
        samples = lhs_sample(N, 
                             ["A", "B", "C", "D", "fr", "cells"],
                             [0, 0, 0, 0, 0.1, 625],
                             [0.1, 0.1, 0.1, 0.1, 0.9, 15625],
                             [False, False, False, False, False, True],
                             seed=42)
        for i in range(N):
            sample = samples[i]
            s, v = sample_games(data_dir=data_dir, exp_dir=experiment_name, runtime=250,
                                a=sample["A"], b=sample["B"], c=sample["C"], d=sample["D"],
                                initialTumor=0, turnover=0.009, init_cells=sample["cells"],
                                prop_res=sample["fr"], radius=2, replicates=1,
                                spaces=["2D"], name=str(i), writeModelFrequency=250)
            run_output += s
            visual_output += v
    elif experiment_name == "HAL3D":
        N = 1200
        samples = lhs_sample(N,
                             ["A", "B", "C", "D", "fr", "cells"],
                             [0, 0, 0, 0, 0.1, 625],
                             [0.1, 0.1, 0.1, 0.1, 0.9, 15625],
                             [False, False, False, False, False, True],
                             seed=42)
        for i in range(N):
            sample = samples[i]
            s, v = sample_games(data_dir=data_dir, exp_dir=experiment_name, runtime=250,
                                a=sample["A"], b=sample["B"], c=sample["C"], d=sample["D"],
                                initialTumor=0, turnover=0.009, init_cells=sample["cells"],
                                prop_res=sample["fr"], radius=2, replicates=1,
                                spaces=["3D"], name=str(i), writeModelFrequency=250)
            run_output += s
            visual_output += v
    else:
        print("Invalid experiment name.")
        exit()
    if len(run_output) > 999:
        run_output = [run_output[i:i + 999] for i in range(0, len(run_output), 999)]
        visual_output = [visual_output[i:i + 999] for i in range(0, len(visual_output), 999)]
        for i in range(len(run_output)):
            write_scripts(data_dir, experiment_name, run_output[i], visual_output[i], i)
    else:
        write_scripts(data_dir, experiment_name, run_output, visual_output)
    print("Make sure you recompile SpatialEGT before pushing jobs:")
    print("javac -d \"build\" -cp \"lib/*\" @sources.txt")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide data directory and experiment name to save model output to.")
    else:
        main(sys.argv[1], sys.argv[2])

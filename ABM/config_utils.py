import json

from scipy.stats import qmc


def write_run_scripts(data_dir, experiment_name, run_output):
    run_output_batches = [run_output[i:i + 999] for i in range(0, len(run_output), 999)]
    for i,batch in enumerate(run_output_batches):
        with open(f"{data_dir}/{experiment_name}/run{i}.sh", "w") as f:
            for output_line in batch:
                f.write(output_line)
    print("Make sure you recompile SpatialEGT before pushing jobs:")
    print("javac -d \"build\" -cp \"lib/*\" @sources.txt")


def write_config(data_dir, exp_dir, config_name, payoff, num_cells, proportion_r,
                 null=1, adaptive=0, continuous=0, write_freq=250, ticks=250, radius=2,
                 turnover=0.009, drug_reduction=0.5, at_threshold=0.5, init_tumor=0, toy_gap=5):
    config = {
        "null": null,
        "adaptive": adaptive,
        "continuous": continuous,
        "writeModelFrequency": write_freq,
        "x": 125,
        "y": 125,
        "neighborhoodRadius": radius,
        "numTicks": ticks,
        "deathRate": turnover,
        "drugGrowthReduction": drug_reduction,
        "numCells": num_cells,
        "proportionResistant": proportion_r,
        "adaptiveTreatmentThreshold": at_threshold,
        "initialTumor": init_tumor,
        "toyGap": toy_gap,
        "A": payoff[0],
        "B": payoff[1],
        "C": payoff[2],
        "D": payoff[3]
    }
    config_path = f"{data_dir}/{exp_dir}/{config_name}/{config_name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def lhs(num_samples, param_names, lower_bounds, upper_bounds, ints, seed):
    sampler = qmc.LatinHypercube(d=len(lower_bounds), seed=seed)
    unscaled_sample = sampler.random(n=num_samples)
    sample = qmc.scale(unscaled_sample, lower_bounds, upper_bounds).tolist()
    sampled_params = [{param_names[i]:round(s[i]) if ints[i] else round(s[i], 2)
                       for i in range(len(s))} for s in sample]
    return sampled_params

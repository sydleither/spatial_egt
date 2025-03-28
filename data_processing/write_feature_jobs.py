import sys

from data_processing.spatial_statistics.custom import nc_dist, proportion_s, sfp_dist
from data_processing.spatial_statistics.muspan import (anni, circularity_dist, cpcf, cross_k,
                                                       entropy, fractal_dimension_dist,
                                                       global_moransi, j_function, kl_divergence,
                                                       local_moransi_dist, nn_dist,
                                                       patch_count, qcm, wasserstein)


FEATURE_REGISTRY = {
    # Custom
    "NC_Resistant": nc_dist,
    "NC_Sensitive": nc_dist,
    "Proportion_Sensitive": proportion_s,
    "SFP": sfp_dist,
    # MuSpAn
    "ANNI_Resistant": anni,
    "ANNI_Sensitive": anni,
    "Entropy": entropy,
    "CPCF_Resistant": cpcf,
    "CPCF_Sensitive": cpcf,
    "Cross_Ripleys_k_Resistant": cross_k,
    "Cross_Ripleys_k_Sensitive": cross_k,
    "Global_Morans_i_Resistant": global_moransi,
    "Global_Morans_i_Sensitive": global_moransi,
    "J_Function_Resistant": j_function,
    "J_Function_Sensitive": j_function,
    "KL_Divergence": kl_divergence,
    "Local_Morans_i_Resistant": local_moransi_dist,
    "Local_Morans_i_Sensitive": local_moransi_dist,
    "NN_Resistant": nn_dist,
    "NN_Sensitive": nn_dist,
    "Patch_Circularity_Resistant": circularity_dist,
    "Patch_Circularity_Sensitive": circularity_dist,
    "Patch_Count_Resistant": patch_count,
    "Patch_Count_Sensitive": patch_count,
    "Patch_Fractal_Dimension_Resistant": fractal_dimension_dist,
    "Patch_Fractal_Dimension_Sensitive": fractal_dimension_dist,
    "SES": qcm,
    "Wasserstein": wasserstein
}

FEATURE_PARAMS = {
    "in_silico": {
        "ANNI_Resistant": {"cell_type1": "resistant", "cell_type2": "sensitive"},
        "ANNI_Sensitive": {"cell_type1": "sensitive", "cell_type2": "resistant"},
        "NC_Resistant": {"radius": 3, "return_fs": True},
        "NC_Sensitive": {"radius": 3, "return_fs": False},
        "SFP": {"sample_length": 5},
        "Cross_Ripleys_k_Resistant": {"max_radius": 6, "step": 1, "cell_type1":"resistant", "cell_type2":"sensitive"},
        "Cross_Ripleys_k_Sensitive": {"max_radius": 6, "step": 1, "cell_type1":"sensitive", "cell_type2":"resistant"},
        "CPCF_Sensitive": {"max_radius": 5, "annulus_step": 1, "annulus_width": 3, "cell_type1":"sensitive", "cell_type2":"resistant"},
        "CPCF_Resistant": {"max_radius": 5, "annulus_step": 1, "annulus_width": 3, "cell_type1":"resistant", "cell_type2":"sensitive"},
        "Global_Morans_i_Resistant": {"cell_type": "resistant", "side_length": 5},
        "Global_Morans_i_Sensitive": {"cell_type": "sensitive", "side_length": 5},
        "J_Function_Resistant": {"cell_type": "resistant", "radius_step": 1},
        "J_Function_Sensitive": {"cell_type": "sensitive", "radius_step": 1},
        "KL_Divergence": {"mesh_step": 3},
        "Local_Morans_i_Resistant": {"cell_type": "resistant", "side_length": 5},
        "Local_Morans_i_Sensitive": {"cell_type": "sensitive", "side_length": 5},
        "NN_Resistant": {"cell_type1": "resistant", "cell_type2": "sensitive"},
        "NN_Sensitive": {"cell_type1": "sensitive", "cell_type2": "resistant"},
        "Patch_Count_Resistant": {"cell_type": "resistant", "alpha": 3, "pad":True},
        "Patch_Count_Sensitive": {"cell_type": "sensitive", "alpha": 3, "pad":True},
        "Patch_Circularity_Resistant": {"cell_type": "resistant", "alpha": 3, "pad":True},
        "Patch_Circularity_Sensitive": {"cell_type": "sensitive", "alpha": 3, "pad":True},
        "Patch_Fractal_Dimension_Resistant": {"cell_type": "resistant", "alpha": 3, "pad":True},
        "Patch_Fractal_Dimension_Sensitive": {"cell_type": "sensitive", "alpha": 3, "pad":True},
        "SES": {"side_length": 10}
    },
    "in_vitro": {
        "ANNI_Resistant": {"cell_type1": "resistant", "cell_type2": "sensitive"},
        "ANNI_Sensitive": {"cell_type1": "sensitive", "cell_type2": "resistant"},
        "NC_Resistant": {"radius": 30, "return_fs": True},
        "NC_Sensitive": {"radius": 30, "return_fs": False},
        "SFP": {"sample_length": 50},
        "Cross_Ripleys_k_Resistant": {"max_radius": 60, "step": 10, "cell_type1":"resistant", "cell_type2":"sensitive"},
        "Cross_Ripleys_k_Sensitive": {"max_radius": 60, "step": 10, "cell_type1":"sensitive", "cell_type2":"resistant"},
        "CPCF_Sensitive": {"max_radius": 50, "annulus_step": 10, "annulus_width": 30, "cell_type1":"sensitive", "cell_type2":"resistant"},
        "CPCF_Resistant": {"max_radius": 50, "annulus_step": 10, "annulus_width": 30, "cell_type1":"resistant", "cell_type2":"sensitive"},
        "Global_Morans_i_Resistant": {"cell_type": "resistant", "side_length": 50},
        "Global_Morans_i_Sensitive": {"cell_type": "sensitive", "side_length": 50},
        "J_Function_Resistant": {"cell_type": "resistant", "radius_step": 10},
        "J_Function_Sensitive": {"cell_type": "sensitive", "radius_step": 10},
        "Local_Morans_i_Resistant": {"cell_type": "resistant", "side_length": 50},
        "Local_Morans_i_Sensitive": {"cell_type": "sensitive", "side_length": 50},
        "NN_Resistant": {"cell_type1": "resistant", "cell_type2": "sensitive"},
        "NN_Sensitive": {"cell_type1": "sensitive", "cell_type2": "resistant"},
        "Patch_Count_Resistant": {"cell_type": "resistant", "alpha": 30},
        "Patch_Count_Sensitive": {"cell_type": "sensitive", "alpha": 30},
        "Patch_Circularity_Resistant": {"cell_type": "resistant", "alpha": 30},
        "Patch_Circularity_Sensitive": {"cell_type": "sensitive", "alpha": 30},
        "Patch_Fractal_Dimension_Resistant": {"cell_type": "resistant", "alpha": 30},
        "Patch_Fractal_Dimension_Sensitive": {"cell_type": "sensitive", "alpha": 30},
        "SES": {"side_length": 100}
    }
}
FEATURE_PARAMS["in_silico_games"] = FEATURE_PARAMS["in_silico"]

FUNCTION_LABELS = {
    "NC_Resistant": {"x":"Fraction Sensitive in Neighborhood", "y":"Fraction of Resistant Cells"},
    "NC_Sensitive": {"x":"Fraction Resistant in Neighborhood", "y":"Fraction of Sensitive Cells"},
    "SFP": {"x":"Fraction Sensitive", "y":"Frequency Across Subsamples"},
    "CPCF_Resistant": {"x":"r", "y":"g(r)"},
    "CPCF_Sensitive": {"x":"r", "y":"g(r)"},
    "Cross_Ripleys_k_Resistant": {"x":"r", "y":"g(r)"},
    "Cross_Ripleys_k_Sensitive": {"x":"r", "y":"g(r)"},
    "J_Function_Resistant": {"x": "r", "y":"j(r)"},
    "J_Function_Sensitive": {"x": "r", "y":"j(r)"},
    "Local_Morans_i_Resistant": {"x":"Moran's i", "y":"Proportion"},
    "Local_Morans_i_Sensitive": {"x":"Moran's i", "y":"Proportion"},
    "NN_Resistant": {"x":"Distance", "y":"Proportion"},
    "NN_Sensitive": {"x":"Distance", "y":"Proportion"},
    "Patch_Circularity_Resistant": {"x":"Patch Circularity", "y":"Proportion"},
    "Patch_Circularity_Sensitive": {"x":"Patch Circularity", "y":"Proportion"},
    "Patch_Fractal_Dimension_Resistant": {"x":"Patch Fractal Dimension", "y":"Proportion"},
    "Patch_Fractal_Dimension_Sensitive": {"x":"Patch Fractal Dimension", "y":"Proportion"}
}

DISTRIBUTION_BINS = {
    "NC_Resistant": (0, 1.1, 0.1),
    "NC_Sensitive": (0, 1.1, 0.1),
    "SFP": (0, 1.1, 0.1),
    "Local_Morans_i_Resistant": (-5, 5.5, 0.5),
    "Local_Morans_i_Sensitive": (-5, 5.5, 0.5),
    "NN_Resistant": (1, 5, 0.2),
    "NN_Sensitive": (1, 5, 0.2),
    "Patch_Circularity_Resisant": (0, 1.1, 0.1),
    "Patch_Circularity_Sensitive": (0, 1.1, 0.1),
    "Patch_Fractal_Dimension_Resistant": (0, 1.1, 0.1),
    "Patch_Fractal_Dimension_Sensitive": (0, 1.1, 0.1)
}


def main(data_type, run_local):
    run_cmd = "python3 -m" if run_local else "sbatch job.sb"
    output = []
    for feature_name in FEATURE_REGISTRY.keys():
        output.append(f"{run_cmd} data_processing.processed_to_feature {data_type} {feature_name}\n")
    with open(f"process_features_{data_type}.sh", "w") as f:
        for output_line in output:
            f.write(output_line)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1], False)
    elif len(sys.argv) == 3:
        main(sys.argv[1], True)
    else:
        print("Please provide the data type and an extra flag if running locally.")

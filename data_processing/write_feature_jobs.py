import sys

import numpy as np

from data_processing.spatial_statistics.custom import nc_dist, proportion_s, sfp_dist
from data_processing.spatial_statistics.muspan import (anni, entropy, cpcf, cross_k,
                                                       j_function, kl_divergence,
                                                       global_moransi, local_moransi_dist,
                                                       nn_dist, qcm, wasserstein,
                                                       circularity, fractal_dimension)


FEATURE_REGISTRY = {
    # Custom
    "NC_(Resistant)": nc_dist,
    "NC_(Sensitive)": nc_dist,
    "Proportion_Sensitive": proportion_s,
    "SFP": sfp_dist,
    # MuSpAn
    "ANNI": anni,
    "Entropy": entropy,
    "CPCF": cpcf,
    "Cross_Ripleys_k": cross_k,
    "J_Function": j_function,
    "KL_Divergence": kl_divergence,
    "Global_Morans_i": global_moransi,
    "Local_Morans_i": local_moransi_dist,
    "NN": nn_dist,
    "SES": qcm,
    "Wasserstein": wasserstein,
    # Temp
    "Patch_Circularity": circularity,
    "Patch_Fractal_Dim": fractal_dimension
}

FEATURE_PARAMS = {
    "in_silico": {
        "NC_(Resistant)": {"radius": 3, "return_fs": True},
        "NC_(Sensitive)": {"radius": 3, "return_fs": False},
        "SFP": {"sample_length": 5},
        "Cross_Ripleys_k": {"max_radius": 5, "step": 1},
        "CPCF": {"max_radius": 5, "annulus_step": 1, "annulus_width": 3},
        "J_Function": {"cell_type": "sensitive", "radius_step": 2},
        "KL_Divergence": {"mesh_step": 3},
        "Global_Morans_i (Resistant)": {"cell_type": "resistant", "side_length": 5},
        "Global_Morans_i (Sensitive)": {"cell_type": "sensitive", "side_length": 5},
        "Local_Morans_i (Resistant)": {"cell_type": "resistant", "side_length": 5},
        "Local_Morans_i (Sensitive)": {"cell_type": "sensitive", "side_length": 5},
        "SES": {"side_length": 10},
        "Patch_Circularity": {"cell_type": "sensitive", "alpha": 3},
        "Patch_Fractal_Dim": {"cell_type": "sensitive", "alpha": 3}
    },
    "in_vitro": {
        "NC_(Resistant)": {"radius": 30, "return_fs": True},
        "NC_(Sensitive)": {"radius": 30, "return_fs": False},
        "SFP": {"sample_length": 50},
        "Cross_Ripleys_k": {"max_radius": 50, "step": 10},
        "CPCF": {"max_radius": 50, "annulus_step": 10, "annulus_width": 30},
        "J_Function": {"cell_type": "sensitive", "radius_step": 20},
        "KL_Divergence": {"mesh_step": 30},
        "Global_Morans_i (Resistant)": {"cell_type": "resistant", "side_length": 50},
        "Global_Morans_i (Sensitive)": {"cell_type": "sensitive", "side_length": 50},
        "Local_Morans_i (Resistant)": {"cell_type": "resistant", "side_length": 50},
        "Local_Morans_i (Sensitive)": {"cell_type": "sensitive", "side_length": 50},
        "SES": {"side_length": 100},
        "Patch_Circularity": {"cell_type": "sensitive", "alpha": 30},
        "Patch_Fractal_Dim": {"cell_type": "sensitive", "alpha": 30}
    }
}


def main(data_type, run_local):
    run_cmd = "python3 -m" if run_local else "sbatch job.sb"
    output = []
    for feature_name in FEATURE_REGISTRY.keys():
        output.append(f"{run_cmd} data_processing.processed_to_feature {feature_name} {data_type}\n")
    with open("process_features.sh", "w") as f:
        for output_line in output:
            f.write(output_line)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1], False)
    elif len(sys.argv) == 3:
        main(sys.argv[1], True)
    else:
        print("Please provide the data type and an extra flag if running locally.")

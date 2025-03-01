import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import skew


def dist_to_hist(dist, bins):
    hist = np.histogram(dist, bins)
    return hist


def get_dist_statistics(name, dist):
    features = dict()
    features[f"{name} Mean"] = np.mean(dist)
    features[f"{name} SD"] = np.std(dist)
    features[f"{name} Skew"] = skew(dist)
    return features


def get_func_statistics(name, func):
    features = dict()
    features[f"{name} 0"] = func[0]
    features[f"{name} Min"] = np.min(func)
    features[f"{name} Max"] = np.max(func)
    return features

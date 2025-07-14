"""Variables and functions used throughout the codebase"""

import os

from pandas.api.types import is_object_dtype


game_colors = {"Sensitive Wins":"#4C956C", "Coexistence":"#9C6D57",
               "Bistability":"#047495", "Resistant Wins":"#EF7C8E"}
theme_colors = ["xkcd:faded purple", "xkcd:yellow orange"]


def get_data_path(data_type:str, data_stage:str, timepoint:int=None):
    """Get the path to the data if it exists, otherwise create the directories

    :param data_type: the name of the directory storing the data
    :type data_type: str
    :param data_stage: the stage of processing the data is at (raw, processed, etc)
    :type data_stage: str
    :param timepoint: the timepoint of the data
    :type data_stage: int
    :return: the full path to the data
    :rtype: str
    """
    if timepoint is not None:
        data_path = f"data/{data_type}/{timepoint}/{data_stage}"
    else:
        data_path = f"data/{data_type}/{data_stage}"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path


def get_spatial_statistic_type(df, spatial_statistic):
    from spatial_database import STATISTIC_REGISTRY
    stat_calculation = STATISTIC_REGISTRY[spatial_statistic]
    if is_object_dtype(df[spatial_statistic]):
        if stat_calculation.__name__.endswith("dist"):
            return "distribution"
        return "function"
    return "value"


def calculate_game(a:float, b:float, c:float, d:float):
    """Get the game quadrant of the payoff matrix

    :param a: type 1's payoff playing against type 1
    :type a: float
    :param b: type 1's payoff playing against type 2
    :type b: float
    :param c: type 2's payoff playing against type 1
    :type c: float
    :param d: type 2's payoff playing against type 2
    :type d: float
    :return: the game quadrant
    :rtype: str
    """
    if a > c and b > d:
        game = "Sensitive Wins"
    elif c > a and b > d:
        game = "Coexistence"
    elif a > c and d > b:
        game = "Bistability"
    elif c > a and d > b:
        game = "Resistant Wins"
    else:
        game = "Unknown"
    return game

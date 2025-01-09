import numpy as np


def fokker_planck(x, n, mu, awm, amw, s):
    if awm == 0:
        fx = s*x
    else:
        fx = (-x + mu*np.log(1-x)+mu*np.log(x)-x*amw/awm +
            np.log(1+x*awm)*(amw + (1+s+amw)*awm)/(awm**2))
    phi = np.log(x*(1-x)/2*n) - 2*n*fx
    return np.exp(-phi)


def calculate_fp_params(row):
    epsilon = 1 - row["a"]
    awm = row["b"] + epsilon - 1
    sm = row["d"] + epsilon - 1
    amw = row["c"] + epsilon - 1 - sm
    row["awm"] = awm
    row["sm"] = sm
    row["amw"] = amw
    return row


def classify_game(awm, amw, sm):
    a = 0
    b = awm
    c = sm+amw
    d = sm
    if a > c and b > d:
        game = "sensitive_wins"
    elif c > a and b > d:
        game = "coexistence"
    elif a > c and d > b:
        game = "bistability"
    elif c > a and d > b:
        game = "resistant_wins"
    else:
        game = "unknown"
    return game
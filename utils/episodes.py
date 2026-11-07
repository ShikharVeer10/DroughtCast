import numpy as np


def drought_episodes(spei, threshold=-1.0, min_months=2):
    runs = count = 0
    for below in np.asarray(spei) <= threshold:
        count = count + 1 if below else 0
        if below and count == min_months:
            runs += 1
    return runs
import numpy as np


# this funtion is to get the relative root mean square error between two vectors: true and est
def compute_RRMS(true, est):
    RRMS = np.sqrt(np.mean(np.power(np.subtract(true, est), 2))) / np.sqrt(np.mean(np.power(true, 2)))
    return RRMS

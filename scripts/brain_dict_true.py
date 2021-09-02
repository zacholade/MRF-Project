"""
Wraps the brain_dict_true.m function in CoverBLIP:
https://github.com/mgolbabaee/CoverBLIP/blob/master/CoverBLIP%20toolbox/data/brain_dict_true.m
Generates a fingerprint per pixel in the t1/t2/pd maps and returns a 3d map of fingerprints.
"""

from typing import List

import matlab
import matlab.engine
import numpy as np


def brain_dict_true(eng,
                    t1s: List[int],
                    t2s: List[int],
                    pd: float,
                    off: List[int] = None,
                    rf_pulses: List[float] = None):
    if rf_pulses is None:
        with open("../RFpulses.npy", "rb") as f:
            rf_pulses = list(np.load(f))[:300]

    if pd == 0.:
        # If t1 and t2 are both 0 then we just return an empty fingerprint.
        return np.full(len(rf_pulses), 0.), 0

    # The default t1s and t2s values are the default values specified in CoverBLIP for dict generation..
    t1s = t1s if t1s is not None else [*range(100, 2000, 40), *range(2200, 6200, 200)]
    t2s = t2s if t2s is not None else [*range(20, 102, 2), *range(110, 200, 4), *range(220, 620, 20)]

    tr = [10 for _ in range(len(rf_pulses))]
    off = off if off is not None else [0]
    # offs = list(np.array([*range(-250, -150, 40), *range(-50, 52, 2), *range(190, 290, 40)])*0)

    d, dict_norm, look_up_table = eng.brain_dict_true(matlab.double(rf_pulses, size=(len(rf_pulses), 1)),
                                                      matlab.double(tr, size=(len(tr), 1)),
                                                      matlab.double(t1s),
                                                      matlab.double(t2s),
                                                      matlab.double(off), nargout=3)

    # https://stackoverflow.com/questions/54199914/access-attribute-of-elements-within-numpy-array
    # Get imaginary part of each item. Discard real as it is 0 when off res is 0.
    d = np.vectorize(lambda x: x.imag)(np.asarray(d)).reshape(len(rf_pulses))
    dict_norm = np.asarray(dict_norm)
    look_up_table = np.asarray(look_up_table)

    return d, dict_norm

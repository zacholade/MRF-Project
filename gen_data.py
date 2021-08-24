"""
Generates the data used for training models.
Uses subprocess daemons.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import List

import matlab
import matlab.engine
import numpy as np

from scripts.brain_dict_true import brain_dict_true
import logging


logger = logging.getLogger('mrf')
logger.setLevel(logging.INFO)
default_logging_format = ''.join([
    "[",
    "%(asctime)s ",
    "%(levelname)-8s ",
    "%(filename)-15s:",
    "%(lineno)3s - ",
    "%(funcName)-20s ",
    "] ",
    "%(message)s",
])

logging.basicConfig(format=default_logging_format)


def convert_to_compact_array(data, label):
    """
    Takes in a 230x230 shape array and converts it to a flattened 1d array with indices
    """
    indices = np.arange(230 * 230).reshape(230, 230)
    t1, t2, pd, dn = label
    m = np.ma.masked_equal(pd, 0)
    t1_masked, t2_masked, pd_masked, indices_masked, dn_masked = \
        np.ma.masked_array(t1, m.mask), np.ma.masked_array(t2, m.mask), \
        np.ma.masked_array(pd, m.mask), np.ma.masked_array(indices, m.mask), \
        np.ma.masked_array(dn, m.mask)
    t1_compressed, t2_compressed, pd_compressed, indices_compressed, dn_compressed = \
        np.ma.compressed(t1_masked), np.ma.compressed(t2_masked), \
        np.ma.compressed(pd_masked), np.ma.compressed(indices_masked), \
        np.ma.compressed(dn_masked)

    fp_compressed = []
    for index in indices_compressed:
        x = int(index // 230)
        y = int(index % 230)
        fp_compressed.append(data[x][y])
    fp_compressed = np.asarray(fp_compressed)
    # recon = np.zeros((230, 230))
    # x = indices_compressed // 230
    # y = indices_compressed % 230
    # recon[x, y] = t1_compressed[np.arange(len(indices_compressed))]

    label = np.asarray([t1_compressed, t2_compressed, pd_compressed, indices_compressed, dn_compressed])
    label = np.transpose(label)
    data = fp_compressed
    return data, label


def get_all_filenames() -> List[str]:
    files = os.listdir("Data/Labels")
    files = [file for file in files if not file.startswith('.')]
    return files


def gen_fp_map(filename):
    with open("Data/RawLabels/" + filename, "rb") as f:
        scan = np.load(f)

    if os.path.isfile("Data/Output/Data/" + filename):
        logger.info(f"Fingerprint already exists for {filename}.")
        return None, None

    logger.info(f"No fingerprint found for: {filename}. Generating fingerprint.")
    scan = np.transpose(scan, axes=(2, 0, 1))
    # Multiply by 1000 to convert to 1000
    t1, t2, pd = scan[0] * 1000, scan[1] * 1000, scan[2]

    eng = matlab.engine.start_matlab()
    eng.addpath("CoverBLIP/CoverBLIP toolbox/data")
    off = [0]
    with open("Data/RFpulses.npy", "rb") as f:
        rf_pulses = list(np.load(f))[:500]

    fingerprint_map = [[] for _ in range(230)]
    dict_norm_map = [[] for _ in range(230)]
    pixel_counter = 0
    for x in range(230):
        for y in range(230):
            t1_p, t2_p, pd_p = t1[x][y], t2[x][y], pd[x][y]
            pixel_counter += 1
            if pixel_counter % 1000 == 0:
                logger.info(f"Generating fingerprint for {filename}. Pixel {pixel_counter} out of {230 * 230}.")
            fingerprint, dict_norm = brain_dict_true(eng, [t1_p], [t2_p], pd_p, off, rf_pulses)
            # fingerprint, dict_norm = np.zeros(1000), 0
            fingerprint_map[x].append(fingerprint)
            dict_norm_map[x].append(dict_norm)

    dict_norm_map = np.asarray(dict_norm_map).reshape(230, 230)
    fingerprint_map = np.array(fingerprint_map).reshape((230, 230, len(rf_pulses)))
    return fingerprint_map, np.asarray([t1, t2, pd, dict_norm_map])


def make_data(filename):
    data, label = gen_fp_map(filename)
    if label is None or data is None:
        return

    data, label = convert_to_compact_array(data, label)
    with open("Data/Output/Data/" + filename, "wb") as f:
        np.save(f, data)

    with open("Data/Output/Labels/" + filename, "wb") as f:
        np.save(f, label)


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=6)
    try:
        maps_to_generate = get_all_filenames()
        pool.map(make_data, maps_to_generate)
    except Exception as e:
        print(type(e))
        print(e)
    finally:
        pool.close()
        pool.join()

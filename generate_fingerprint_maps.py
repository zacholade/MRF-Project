# Imports
from __future__ import annotations

import os
import multiprocessing

import matlab
import matlab.engine
import numpy as np
from typing import List, Tuple

from scripts.brain_dict_true import brain_dict_true
import matplotlib.pyplot as plt


def get_all_file_paths() -> List[Tuple[str, str, str, str]]:
    """
    Returns a tuple of:
    (PATH_TO_PARAMETER_MAP, PARAMETER_FILE_NAME , PATH_TO_EXACT_FINGERPRINT_MAP, FINGERPRINT_FILE_NAME)
    """
    parameter_folders = ["Data/MRF_maps/ParameterMaps/Train/", "Data/MRF_maps/ParameterMaps/Test/"]
    parameter_files = []
    for parameter_folder in parameter_folders:
        files = os.listdir(parameter_folder)
        files = [file for file in files if not file.startswith('.')]
        for file in files:
            parameter_files.append((parameter_folder, file))

    # Now add path to save for each file.
    all_paths_and_files = []
    for parameter_folder, parameter_file in parameter_files:
        fingerprint_folder = parameter_folder.replace("ParameterMaps", "ExactFingerprintMaps")
        fingerprint_file = parameter_file.replace(".npy", ".npz")
        all_paths_and_files.append((parameter_folder, parameter_file, fingerprint_folder, fingerprint_file))

    return all_paths_and_files


def gen_and_save_fingerprint_map(file_tuple):
    print(file_tuple)
    parameter_path, parameter_file, fingerprint_path, fingerprint_file = file_tuple
    with open(parameter_path + parameter_file, "rb") as f:
        scan = np.load(f)

    if os.path.isfile(fingerprint_path + fingerprint_file):
        print(f"Fingerprint already exists for {fingerprint_file}.")
        return

    print(f"No fingerprint found for: {parameter_path + parameter_file}. Generating fingerprint.")
    scan = np.transpose(scan, axes=(2, 0, 1))
    # Multiply by 1000 to convert to 1000
    t1, t2, pd = scan[0] * 1000, scan[1] * 1000, scan[2]

    eng = matlab.engine.start_matlab()
    eng.addpath("CoverBLIP/CoverBLIP toolbox/data")

    off = [0]
    with open("Data/RFpulses.npy", "rb") as f:
        rf_pulses = list(np.load(f))[:1000]

    fingerprint_map = [[] for i in range(230)]
    pixel_counter = 0
    for x in range(230):
        for y in range(230):
            t1_p, t2_p, pd_p = t1[x][y], t2[x][y], pd[x][y]
            pixel_counter += 1
            if pixel_counter % 1000 == 0:
                print(f"Generating fingerprint for {parameter_file}. Pixel {pixel_counter} out of {230 * 230}.")
            fingerprint = brain_dict_true(eng, [t1_p], [t2_p], pd_p, off, rf_pulses)
            fingerprint_map[x].append(fingerprint)

    fingerprint_map = np.array(fingerprint_map).reshape((230, 230, len(rf_pulses)))

    with open(fingerprint_path + fingerprint_file, "wb") as f:
        print(f"Saving {fingerprint_path + fingerprint_file}")
        np.savez_compressed(f, fingerprint_map)
        print(f"Saved {fingerprint_path + fingerprint_file} successfully.")


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=6)
    try:
        maps_to_generate = get_all_file_paths()
        pool.map(gen_and_save_fingerprint_map, maps_to_generate)
    except Exception as e:
        print(type(e))
        print(e)
    finally:
        pool.close()
        pool.join()

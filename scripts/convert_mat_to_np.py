import numpy as np
import h5py
import scipy.io


def convert_mat_files_to_np_arrays_and_save():
    root = "coverblip_export_res0/"

    # open the files
    dict_mat = h5py.File(root + "dict.mat")
    lut_mat = scipy.io.loadmat(root + "lut.mat")

    # Convert to np arrays
    dict_mat = np.array(dict_mat['dict'])
    lut_mat = np.array(lut_mat['lut'])

    with open("dict.npy", "wb") as f:
        np.save(f, dict_mat)

    with open("lut.npy", "wb") as f:
        np.save(f, lut_mat)


if __name__ == "__main__":
    convert_mat_files_to_np_arrays_and_save()
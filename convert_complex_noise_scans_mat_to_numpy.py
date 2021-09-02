import numpy as np
import h5py
import scipy.io
import re
import os


def convert_to_compact_array(data, label):
    """
    Takes in a 230x230 shape array and converts it to a flattened 1d array with indices
    """
    t1, t2, pd, indices, dn = np.transpose(label, axes=(2, 0, 1))
    m = np.ma.masked_equal(pd, 0)
    t1_masked, t2_masked, pd_masked, indices_masked, dn_masked = \
        np.ma.masked_array(t1, m.mask), np.ma.masked_array(t2, m.mask), \
        np.ma.masked_array(pd, m.mask), np.ma.masked_array(indices, m.mask), \
        np.ma.masked_array(dn, m.mask)
    t1_compressed, t2_compressed, pd_compressed, indices_compressed, dn_compressed = \
        np.ma.compressed(t1_masked), np.ma.compressed(t2_masked), \
        np.ma.compressed(pd_masked), np.ma.compressed(indices_masked), \
        np.ma.compressed(dn_masked)

    x = (indices_compressed // 230).astype(int)
    y = (indices_compressed % 230).astype(int)
    fp_compressed = data[x, y]
    # recon = np.zeros((230, 230))
    # x = indices_compressed // 230
    # y = indices_compressed % 230
    # recon[x, y] = t1_compressed[np.arange(len(indices_compressed))]

    label = np.asarray([t1_compressed, t2_compressed, pd_compressed, indices_compressed, dn_compressed])
    label = np.transpose(label)
    data = fp_compressed
    return data, label


def convert_mat_files_to_np_arrays_and_save():
    root = f"CoverBLIP/CoverBLIP toolbox/data/reconstructed_scans/"
    for filename in os.listdir(root):
        print(filename)
        mat = scipy.io.loadmat(root + filename)

        scan = np.array(mat.get('out_matrix'))

        import matplotlib.pyplot as plt
        out_filename = f"{filename.split('_snr')[0]}.npy"
        no_noise = np.load("Data/Uncompressed/Test/Data/" + out_filename)[:, :, :300]
        # plt.plot(np.arange(300), no_noise[100, 100])
        # plt.plot(scan[100, 100])
        # plt.show()

        new_sum_squares = np.sum(scan**2, axis=2)  # To assert sum of squares == 1
        # print(new_sum_squares)

        snr = re.findall('snr-([0-9]*)', filename)[0]
        sub_sample_ratio = re.findall('cs-([0-9]*)', filename)[0]

        uncompressed_base_output = 'Data/Uncompressed/Test/ComplexNoise'
        if not os.path.exists(uncompressed_base_output):
            os.mkdir(uncompressed_base_output)

        output_dir = f'{uncompressed_base_output}/snr-{snr}_cs-{sub_sample_ratio}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        out_filename = f"{filename.split('_snr')[0]}.npy"
        with open(f"{output_dir}/{out_filename}", "wb") as f:
            np.save(f, scan)

        # Compressed Data: ---------------------------------------------------------

        label = np.load(f"Data/Uncompressed/Test/Labels/{filename.split('_snr')[0]}.npy")
        data, label = convert_to_compact_array(scan, label)

        compressed_base_output = 'Data/Compressed/Test/ComplexNoise'
        if not os.path.exists(compressed_base_output):
            os.mkdir(compressed_base_output)

        output_dir = f'{compressed_base_output}/snr-{snr}_cs-{sub_sample_ratio}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with open(f"{output_dir}/{out_filename}", "wb") as f:
            np.save(f, data)


if __name__ == "__main__":
    convert_mat_files_to_np_arrays_and_save()



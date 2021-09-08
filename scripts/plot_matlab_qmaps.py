import numpy as np
import h5py
import scipy.io
from util import plot_maps


def mape(pred, true):
    return np.mean(np.abs(((true - pred) / true))) * 100

if __name__ == "__main__":
    subj = "subj2_fisp_slc8_23"
    cs = 1
    algo = "BLIP"
    file_name = f"CS{cs}_{algo}_{subj}.mat"
    label = np.load(f"Data/Uncompressed/Test/Labels/{subj}.npy")
    mat = scipy.io.loadmat(f"../CoverBLIP/CoverBLIP toolbox/data/qmaps/{file_name}")
    true_t1, true_t2, true_pd, pos, dn = label[:, :, 0], label[:, :, 1], label[:, :, 2], label[:, :, 3], label[:, :, 4]
    predicted = mat.get('qmap')[:, :, 0:2]
    predicted[true_pd == 0] = 0
    plot_maps(predicted, label[:, :, 0:2], None, epoch=0,
        save_dir="../CoverBLIP/CoverBLIP toolbox/data/qmaps/plots", subj_name=f"CS{cs}_{algo}_{subj}.npy")


    m = np.ma.masked_equal(true_pd, 0)
    pred_t1_masked, pred_t2_masked, true_t1_masked, true_t2_masked = np.ma.masked_array(predicted[:, :, 0], m.mask), np.ma.masked_array(predicted[:, :, 1], m.mask), \
                                                                     np.ma.masked_array(label[:, :, 0], m.mask), np.ma.masked_array(label[:, :, 1], m.mask)
    pred_t1_compressed, pred_t2_compressed, true_t1_compressed, true_t2_compressed = np.ma.compressed(pred_t1_masked), np.ma.compressed(pred_t2_masked), \
                                                                                     np.ma.compressed(true_t1_masked), np.ma.compressed(true_t2_masked)

    t1_mape = mape(pred_t1_compressed, true_t1_compressed)
    t2_mape = mape(pred_t2_compressed, true_t2_compressed)
    print(f"T1 MAPE {t1_mape}")
    print(f"T2 MAPE {t2_mape}")
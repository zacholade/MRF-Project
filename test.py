from util import get_all_data_files
import numpy as np

data_file_names, label_file_names = get_all_data_files("Train")
for label_file_name in label_file_names:
    file = np.load(label_file_name)
    transposed = np.transpose(file)

    with open(label_file_name, "wb") as f:
        np.save(f, transposed)


data_file_names, label_file_names = get_all_data_files("Test")
for label_file_name in label_file_names:
    file = np.load(label_file_name)
    transposed = np.transpose(file)

    with open(label_file_name, "wb") as f:
        np.save(f, transposed)
import numpy as np
import os

file_names = os.listdir("Data/Compressed/Data")
file_names = [file_name for file_name in file_names if not file_name.startswith(".")]

print(file_names)
for file_name in file_names:
    data = np.load("Data/Compressed/Data/" + file_name)
    label = np.load("Data/Compressed/Labels/" + file_name)

    pos = label[:, 3]
    x = (pos // 230).astype(int)
    y = (pos % 230).astype(int)

    label = np.delete(label, 3, axis=1)

    new_data = np.zeros((230, 230, 1000))
    new_label = np.zeros((230, 230, 4))
    new_data[x, y] = data
    new_label[x, y] = label

    with open("Data/Uncompressed/Data/" + file_name, "wb") as f:
        np.save(f, new_data)

    with open("Data/Uncompressed/Labels/" + file_name, "wb") as f:
        np.save(f, new_label)


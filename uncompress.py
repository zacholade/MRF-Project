import numpy as np
import os

file_names = os.listdir("Data/Compressed/Test/Data")
file_names = [file_name for file_name in file_names if not file_name.startswith(".")]

print(file_names)
for file_name in file_names:
    data = np.load("Data/Compressed/Test/Data/" + file_name)
    label = np.load("Data/Compressed/Test/Labels/" + file_name)

    pos = label[:, 3]
    x = (pos // 230).astype(int)
    y = (pos % 230).astype(int)

    new_data = np.zeros((230, 230, 1000))
    new_label = np.zeros((230, 230, 5))
    new_data[x, y] = data
    new_label[x, y] = label

    with open("Data/Uncompressed/Test/Data/" + file_name, "wb") as f:
        np.save(f, new_data)

    with open("Data/Uncompressed/Test/Labels/" + file_name, "wb") as f:
        np.save(f, new_label)


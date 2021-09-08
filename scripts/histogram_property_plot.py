import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict


file_names = os.listdir("Data/Uncompressed/Labels/")
file_names = [file for file in file_names if file.endswith(".npy")]

t1_histogram = defaultdict(int)
t2_histogram = defaultdict(int)
i = 0

max_t1 = 0
max_t2 = 0
for file_name in file_names:
    file = np.load(f"Data/Uncompressed/Labels/{file_name}")
    num_4000 = 0
    for x in range(230):
        for y in range(230):
            t1, t2, pd, _, _ = file[x, y]
            if pd == 0:
                continue

            if t2 >= 2000:
                t2 = 2000

            if t1 > max_t1:
                max_t1 = t1
                print(f"max_t1 {max_t1}")

            if t2 > max_t2:
                max_t2 = t2
                print(f"max t2 {max_t2}")

            if t2 > 2000:
                num_4000 += 1

            i += 1
            t1 = round(t1 / 100) * 100
            t2 = round(t2 / 20) * 10

            t1_histogram[t1] += 1
            t2_histogram[t2] += 1

print('final')
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
fig.tight_layout()
plt.margins(0)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_ylabel('T1 Frequency', family='Arial', fontsize=15)
ax[1].set_ylabel('T2 Frequency', family='Arial', fontsize=15)
ax[0].set_xlabel('T1 value (ms)', family='Arial', fontsize=15)
ax[1].set_xlabel('T2 value (ms)', family='Arial', fontsize=15)
ax[0].set_xlim((0, 6000))
ax[1].set_xlim((0, 2000))
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)
ax[0].bar(t1_histogram.keys(), t1_histogram.values(), width=20)
ax[1].bar(t2_histogram.keys(), t2_histogram.values(), width=5)
ax[0].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
plt.show()

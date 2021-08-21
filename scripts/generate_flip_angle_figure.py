"""
Code used to generate a figure in my dissertation that illustrates
the flip angles and different fingerprint trajectories using the
IR-BSSFP sequence, constant TR of 10 and off-res set to 0
"""

import matlab
import matlab.engine

eng = matlab.engine.start_matlab()
from brain_dict_true import brain_dict_true
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pylab as pl


rf_pulses = list(np.load("Data/RFpulses.npy"))

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.plot(rf_pulses)
ax.set_ylabel("Flip angles (radians)", family='Arial', fontsize=15)
ax.set_xlabel("Excitation number", family='Arial', fontsize=15)
plt.margins(0)
# plt.grid()
ax.spines['right'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0, 1.2])
ax.set_xlim([0, 1000])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


eng.addpath("CoverBLIP/CoverBLIP toolbox/data")

t1s = np.flip(np.arange(100, 3050, 100))
print(t1s)
t2 = 100
pd = 100
off = 0

out_fp = [[] for _ in range(len(t1s))]
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# with open("out_csv.csv", 'a', newline='') as file:
my_cmap = cm.jet
colors = pl.cm.jet(np.linspace(0, 1, len(t1s)))

sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=3000))
cbar = plt.colorbar(sm)
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_ylabel("T1 (ms)", labelpad=15, family='Arial', fontsize=15)
ax.set_ylabel("Normalised fingeprint (a.u.)", family='Arial', fontsize=15)
ax.set_xlabel("Excitation number", family='Arial', fontsize=15)
plt.margins(0)
plt.grid()#linewidth=0.1, color='black')
ax.spines['right'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_color('grey')
ax.spines['top'].set_color('grey')
ax.set_ylim([0, 0.35])
ax.set_xlim([0, 1000])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

for i, t1 in enumerate(t1s):
    print(t1)
    fingerprint, dict_norm = brain_dict_true(eng, [t1], [t2], pd, [off], rf_pulses)
    fingerprint *= dict_norm
    new_dict_norm = np.sqrt(np.sum(np.abs(np.square(fingerprint)), axis=0))  # Calculate new normalisation value per fp.
    fingerprint /= new_dict_norm  # Apply normalisation value to data
    ax.plot(np.abs(fingerprint), color=colors[i], linewidth=1)

plt.show()


#_----------------------------------


t1 = 1000
t2s = np.flip(np.arange(10, 310, 10))
print(t2s)
pd = 100
off = 0

out_fp = [[] for _ in range(len(t2s))]
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# with open("out_csv.csv", 'a', newline='') as file:
my_cmap = cm.jet
colors = pl.cm.jet(np.linspace(0, 1, len(t2s)))

sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=300))
cbar = plt.colorbar(sm)
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_ylabel("T2 (ms)", labelpad=10, family='Arial', fontsize=15)
ax.set_ylabel("Normalised fingeprint (a.u.)", family='Arial', fontsize=15)
ax.set_xlabel("Excitation number", family='Arial', fontsize=15)
plt.margins(0)
plt.grid()
ax.spines['right'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_color('grey')
ax.spines['top'].set_color('grey')
ax.set_ylim([0, 0.30])
ax.set_xlim([0, 1000])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

for i, t2 in enumerate(t2s):
    print(t2)
    fingerprint, dict_norm = brain_dict_true(eng, [t1], [t2], pd, [off], rf_pulses)
    fingerprint *= dict_norm
    new_dict_norm = np.sqrt(np.sum(np.abs(np.square(fingerprint)), axis=0))  # Calculate new normalisation value per fp.
    fingerprint /= new_dict_norm  # Apply normalisation value to data
    ax.plot(np.abs(fingerprint), color=colors[i], linewidth=1)

plt.show()
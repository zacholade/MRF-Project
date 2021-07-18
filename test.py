from multiprocessing import Process
import random
import time
import numpy as np


def some_function():
    file = np.load("Data/MRF_maps/Data/Test/subj8_fisp_slc1_1.npy",
                   mmap_mode='r')
    print(file[4][5])
    time.sleep(100)
    print(file)

if __name__ == '__main__':
    processes = []

    for m in range(1,20):

       p = Process(target=some_function)
       p.start()
       processes.append(p)

    for p in processes:
       p.join()
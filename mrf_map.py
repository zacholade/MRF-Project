from typing import List

import numpy as np


class MRFMap:
    def __init__(self, t1_map, t2_map, pd_map, fingerprint_map):
        self.t1_map = t1_map
        self.t2_map = t2_map
        self.pd_map = pd_map
        self.fingerprint_map = fingerprint_map
        assert self.t1_map.shape == self.t2_map.shape == self.pd_map.shape
        self.mask_t1_map = np.ma.masked_not_equal(t1_map, 0)
        self.mask_t2_map = np.ma.masked_not_equal(t2_map, 0)
        self.mask = np.logical_and(self.mask_t1_map, self.mask_t2_map)

    @property
    def shape_x(self):
        assert self.t1_map.shape == self.t2_map.shape == self.pd_map.shape
        return self.t1_map.shape[0]

    @property
    def shape_y(self):
        assert self.t1_map.shape == self.t2_map.shape == self.pd_map.shape
        return self.t1_map.shape[1]

    @property
    def shape_z(self):
        return 1000  # Always working with fingerprints with 1000 timepoints.

    def __iter__(self):
        """
        Iterates through t1/t2/pd/fp on a per pixel basis.
        :return: t1_p, t2_p, pd_p, fp_p
        """
        x_length, y_length = self.t1_map.shape
        for x in range(x_length):
            for y in range(y_length):
                t1_p, t2_p, pd_p, fp_p = self.t1_map[x][y], self.t2_map[x][y], self.pd_map[x][y], self.fingerprint_map[x][y]
                if t1_p == 0 and t2_p == 0:
                    continue
                yield t1_p, t2_p, pd_p, fp_p

    @classmethod
    def from_scan_and_fp(cls, scan, fp):
        scan = np.transpose(scan, axes=(2, 0, 1))
        # Multiply by 1000 to convert to 1000
        t1, t2, pd = scan[0] * 1000, scan[1] * 1000, scan[2]
        return cls(t1, t2, pd, fp)

# -*- coding: utf-8 -*-
import time
import numpy as np
from utils.utils import euclidean_dist, gaussian_fun

class GaussianKernelModel(object):
    def __init__(self):
        self.avg_dist = None
        self.all_psi = None
        self.check_in_matrix = None
        self.poi_coos = None

    def save_result(self, path):
        ctime = time.time()   
        print("Saving GaussianKernelModel the result...")
        np.save(path + "avg_dist", self.avg_dist)
        np.save(path + "all_psi", self.all_psi)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_result(self, path):
        ctime = time.time()
        print("Loading GaussianKernelModel result...",)
        self.avg_dist = np.load(path + "avg_dist.npy")
        self.all_psi = np.load(path + "all_psi.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def pre_compute_all_prob(self, check_in_matrix, poi_coos, user_home):
        self.poi_coos = poi_coos
        self.check_in_matrix = check_in_matrix
        ctime = time.time()
        print("Training GeographyModel...", )
        avg_dist = []
        uid_psi = []
        for uid in range(check_in_matrix.shape[0]):
            uid_lids = check_in_matrix[uid, :].nonzero()[0]
            uid_distance = []
            uid_tohome_dist = []
            for i in uid_lids:
                d = float(euclidean_dist(user_home[uid], poi_coos[i]))
                uid_tohome_dist.append(d)
                for j in range(check_in_matrix.shape[1]):
                    coo1, coo2 = poi_coos[i], poi_coos[j]
                    distance = float(euclidean_dist(coo1, coo2))
                    uid_distance.append(distance)
            uid_max_dist = np.max(np.array(uid_tohome_dist))
            uid_avg_dist = gaussian_fun(uid_distance, uid_max_dist)
            uid_psi.append(uid_max_dist)
            avg_dist.append(uid_avg_dist)
        self.all_psi = np.array(uid_psi)
        self.avg_dist = np.array(avg_dist)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, uid, lj, check_in_matrix, poi_coos, user_home):
        uid_lids = check_in_matrix[uid, :].nonzero()[0]
        todist = []
        for i in uid_lids:
            dist1 = float(euclidean_dist(poi_coos[i], poi_coos[lj]))
            todist.append(dist1)
        pro = gaussian_fun(todist, self.all_psi[uid]) / self.avg_dist[uid]
        pro = (1.0 / (float(euclidean_dist(user_home[uid], poi_coos[lj])) + 1)) * pro
        return pro

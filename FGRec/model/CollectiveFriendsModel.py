# -*- coding: utf-8 -*-
import time
import numpy as np
import random

class CollectiveFriendsModel(object):
    def __init__(self, lamb):
        self.uid_lid_list = None
        self.lamb = lamb
        self.w_1 = np.array([random.uniform(0.0, 1.0) for i in range(4)])
        self.w_2 = np.array([random.uniform(0.0, 1.0) for i in range(4)])
        self.para = np.array([0.0, 0.0, 0.0, 0.0])

    def save_result(self, path):
        ctime = time.time()
        print("Saving the CFM result...")
        np.save(path + "CFM", self.para)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_result(self, path, NF, SF):
        ctime = time.time()
        print("Loading CFM result...")
        self.para = np.load(path + "CFM.npy")
        self.NF = NF
        self.SF = SF
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def sgd_lns(self, max_iters, uid_lid_list, user_features, NF, SF):
        print("Training CFModel...")
        self.NF = NF
        self.SF = SF
        uid_lid_list = list(uid_lid_list)

        def sigmoid(x):
            return 1.0 / (1 + np.exp(-x))

        w_1 = self.w_1
        w_2 = self.w_2
        lamb = self.lamb
        learn_rate = 0.005
        for iters in range(max_iters):
            random.shuffle(uid_lid_list)
            for uid, lid in uid_lid_list:
                phi_1 = sigmoid(w_1.dot(user_features[uid][2:6].T))
                phi_2 = sigmoid(w_2.dot(user_features[uid][7:].T))

                d_w1 = 2 * lamb * w_1 - phi_1 * (1 - phi_1) * user_features[uid][2:6] * NF[uid, lid]
                d_w2 = 2 * lamb * w_2 - phi_2 * (1 - phi_2) * user_features[uid][7:] * SF[uid, lid]

                w_1 = w_1 - learn_rate * d_w1
                w_2 = w_2 - learn_rate * d_w2

        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.para = np.array([phi_1, phi_2])
        print("Done CFMModel...")
        return phi_1, phi_2

    def predict(self, uid, lid):
        phi_1 = self.para[0]
        phi_2 = self.para[1]
        return phi_1 * self.NF[uid][lid] + phi_2 * self.SF[uid][lid]

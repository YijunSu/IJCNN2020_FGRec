# -*- coding: utf-8 -*-
import time
import numpy as np

class JointPoissonFactorModel(object):
    def __init__(self, K=50, alpha=20.0, beta=0.2):
        self.K = K
        self.alpha_U = 30.0
        self.alpha_L = 30.0
        self.alpha_Z = 30.0
        self.beta = beta
        self.U, self.L, self.Z = None, None, None

    def save_model(self, path):
        ctime = time.time()
        print("Saving U and L and Z...",)
        np.save(path + "U", self.U)
        np.save(path + "L", self.L)
        np.save(path + "Z", self.Z)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_model(self, path):
        ctime = time.time()
        print("Loading U and L and Z...",)
        self.U = np.load(path + "U.npy")
        self.L = np.load(path + "L.npy")
        self.Z = np.load(path + "Z.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def train(self, sparse_user_poi_matrix, sparse_user_cate_matrix, max_iters=10, learning_rate=1e-4):
        ctime = time.time()
        print("Training JPF...", )
        alpha_U = self.alpha_U 
        alpha_L = self.alpha_L 
        alpha_Z = self.alpha_Z 
        beta = self.beta
        K = self.K

        C_x = sparse_user_poi_matrix
        C_y = sparse_user_cate_matrix
        M, N = sparse_user_poi_matrix.shape
        M, P = sparse_user_cate_matrix.shape

        U = 0.5 * np.sqrt(np.random.gamma(alpha_U, beta, (M, K))) / K
        L = 0.5 * np.sqrt(np.random.gamma(alpha_L, beta, (N, K))) / K
        Z = 0.5 * np.sqrt(np.random.gamma(alpha_Z, beta, (P, K))) / K 

        C_x = C_x.tocoo()
        entry_index_x = list(zip(C_x.row, C_x.col))
        C_x = C_x.tocsr()
        C_x_dok = C_x.todok()

        C_y = C_y.tocoo() 
        entry_index_y = list(zip(C_y.row, C_y.col))
        C_y = C_y.tocsr()
        C_y_dok = C_y.todok()

        tau = 10
        last_loss = float('Inf')
        for iters in range(max_iters):
            C_x_Y = C_x_dok.copy()
            for i, j in entry_index_x:
                C_x_Y[i, j] = 1.0 * C_x_dok[i, j] / U[i].dot(L[j]) - 1 #(10)
            C_x_Y = C_x_Y.tocsr()

            C_y_Y = C_y_dok.copy()
            for i, l in entry_index_y:
                C_y_Y[i, l] = 1.0 * C_y_dok[i, l] / U[i].dot(Z[l]) - 1 #(10)
            C_y_Y = C_y_Y.tocsr()

            learning_rate_k = learning_rate * tau / (tau + iters - 1)
            U += learning_rate_k * (C_x_Y.dot(L) + C_y_Y.dot(Z) + (alpha_U - 1) / U - 1 / beta)
            L += learning_rate_k * ((C_x_Y.T).dot(U) + (alpha_L - 1) / L - 1 / beta)
            Z += learning_rate_k * ((C_y_Y.T).dot(U) + (alpha_Z - 1) / Z - 1 / beta)

            loss = 0.0
            loss1 = 0.0
            for i, j in entry_index_x:
                loss1 += (C_x_dok[i, j] - U[i].dot(L[j]))**2
            print('Iteration:', iters,  'loss1:', loss1)

            loss2 = 0.0
            for i, l in entry_index_y:
                loss2 += (C_y_dok[i, l] - U[i].dot(Z[l]))**2
            print('Iteration:', iters,  'loss2:', loss2)
            
            loss = loss1 + loss2
            print('Iteration:', iters,  'loss:', loss)

            if loss > last_loss:
                print("Early termination.")
                break
            last_loss = loss

        print("Done. Elapsed time:", time.time() - ctime, "s")
        self.U, self.L, self.Z = U, L, Z

    def predict(self, uid, lid, sigmoid, POI_cate_matrix, delta):
        def normalize(scores):
            if (len(scores)==0):
                scores = [0.01]
            max_score = max(scores)
            if not max_score == 0:
                scores = [s / max_score for s in scores]
            return scores

        if sigmoid:
            ca = 1.0 / (1+len(POI_cate_matrix[lid, :].nonzero()[0]))
            score_uz = np.sum(normalize([self.U[uid].dot(self.Z[c]) for c in POI_cate_matrix[lid, :].nonzero()[0]]))
            score = delta + ca * score_uz
            f_socre = score * self.U[uid].dot(self.L[lid])
            return 1.0 / (1 + np.exp(-f_socre))
        return (delta + 1.0/len(POI_cate_matrix[lid, :].nonzero()[0]) * np.sum([self.U[uid].dot(self.Z[c]) for c in POI_cate_matrix[lid, :].nonzero()[0]])) * self.U[uid].dot(self.L[lid])
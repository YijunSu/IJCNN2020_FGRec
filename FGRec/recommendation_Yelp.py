# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sparse
import time
from collections import defaultdict
from sklearn import preprocessing
from model.JointPoissonFactorModel import JointPoissonFactorModel
from model.CollectiveFriendsModel import CollectiveFriendsModel
from model.CFM_SFModel import CFMSFModel
from model.CFM_NFModel import CFMNFModel
from model.GaussianKernelModel import GaussianKernelModel
from metric.metrics import precisionk, recallk, mapk, ndcgk


def read_training_data():
    train_data = open(train_file, 'r').readlines()
    sparse_training_matrix = sparse.dok_matrix((num_users, num_pois))
    user_poi_matrix = np.zeros((num_users, num_pois))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparse_training_matrix[uid, lid] = freq
        user_poi_matrix[uid, lid] = 1.0
        # user_poi_matrix[uid, lid] = freq
        training_tuples.add((uid, lid))
    return sparse_training_matrix, training_tuples, user_poi_matrix


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_friends = open(user_social_friends_file, 'r').readlines()
    neighbor_friends = open(user_neighbor_friends_file, 'r').readlines()
    u_vec = open(user_vector_file, 'r').readlines()
    social_matrix = np.zeros((num_users, num_users))
    user_social_friends = defaultdict(list)
    user_neighbor_friends = defaultdict(list)
    user_vectors = defaultdict(list)

    for eachline in social_data:
        uid1, uid2 = eachline.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        social_matrix[uid1, uid2] = 1.0
        social_matrix[uid2, uid1] = 1.0

    for eachline in social_friends:
        users = eachline.strip().split()
        user_id = users[0]
        for u in users[1:]:
            user_social_friends[user_id].append(u)

    for eachline in neighbor_friends:
        users = eachline.strip().split()
        user_id = users[0]
        for u in users[1:]:
            user_neighbor_friends[user_id].append(u)

    for eachline in u_vec:
        u_vectors = eachline.strip().split()
        user_id = u_vectors[0]
        for vec in u_vectors[1:]:
            user_vectors[user_id].append(float(vec))
    return social_matrix, user_social_friends, user_neighbor_friends, user_vectors


def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos


def read_user_homes():
    home_coos = {}
    home_data = open(user_home_file, 'r').readlines()
    for eachline in home_data:
        uid, lat, lng = eachline.strip().split()
        uid, lat, lng = int(uid), float(lat), float(lng)
        home_coos[uid] = (lat, lng)
    return home_coos


def read_category_data():
    category_data = open(category_file, 'r').readlines()
    sparse_cate_matrix = sparse.dok_matrix((num_users, num_categories))
    for eachline in category_data:
        uid, cid, freq = eachline.strip().split()
        uid, cid, freq = int(uid), int(cid), int(freq)
        sparse_cate_matrix[uid, cid] = freq
    return sparse_cate_matrix


def read_POI_category():
    poi_category_data = open(POI_category_file, 'r').readlines()
    POI_cate_matrix = np.zeros((num_pois, num_categories))
    for eachline in poi_category_data:
        lid, cid = eachline.strip().split()
        lid, cid, = int(lid), int(cid)
        POI_cate_matrix[lid, cid] = 1.0
    return POI_cate_matrix


def read_user_feature():
    feature_data = np.loadtxt(user_features_file)
    feature_data_normalized1 = preprocessing.normalize(feature_data[:, 2:6], norm='l2')
    feature_data_normalized2 = preprocessing.normalize(feature_data[:, 7:], norm='l2')
    feature_data1 = np.column_stack((feature_data[:, :2], feature_data_normalized1, feature_data[:, 6], feature_data_normalized2))
    return feature_data1


def normalize(scores):
    max_score = max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores


def main():
    sparse_training_matrix, training_tuples, user_poi_matrix = read_training_data()
    social_matrix, user_social_friends, user_neighbor_friends, user_vectors = read_friend_data()
    ground_truth = read_ground_truth()
    user_features = read_user_feature()
    poi_coos = read_poi_coos()
    home_coos = read_user_homes()
    user_cate_matrix = read_category_data()
    POI_cate_matrix = read_POI_category()

    start_time = time.time()
    # JPF.train(sparse_training_matrix, user_cate_matrix, max_iters=20, learning_rate=1e-5)
    # JPF.save_model("./tmp/")
    JPF.load_model("./tmp/")

    # CFM_SF.compute_friend_sim(user_social_friends, user_social_friends, user_poi_matrix, user_vectors)
    # SF = CFM_SF.compute_pro_matrix(num_users, num_pois)
    # CFM_SF.save_result("./tmp/", "SF")
    CFM_SF.load_result("./tmp/", "SF")

    # CFM_NF.compute_friend_sim(user_neighbor_friends, user_social_friends, user_poi_matrix)
    # NF = CFM_NF.compute_pro_matrix(num_users, num_pois)
    # CFM_NF.save_result("./tmp/", "NF")
    CFM_NF.load_result("./tmp/", "NF")

    # CFM.sgd_lns(100, training_tuples, user_features, CFM_NF.pro_matrix, CFM_SF.pro_matrix)
    # CFM.save_result("./tmp/")
    CFM.load_result("./tmp/", CFM_NF.pro_matrix, CFM_SF.pro_matrix)

    # GKM.pre_compute_all_prob(user_poi_matrix, poi_coos, home_coos)
    # GKM.save_result("./tmp/")
    GKM.load_result("./tmp/")

    result_p = open("./result/result_top_" + "Pre" + "_FGRec_Y.txt", 'w')
    result_r = open("./result/result_top_" + "Rec" + "_FGRec_Y.txt", 'w')
    result_m = open("./result/result_top_" + "MAP" + "_FGRec_Y.txt", 'w')
    result_n = open("./result/result_top_" + "NDCG" + "_FGRec_Y.txt", 'w')
    elapsed_time = time.time() - start_time
    print("Done. Elapsed time:", elapsed_time, "s")

    all_uids = list(range(num_users))
    all_lids = list(range(num_pois))
    np.random.shuffle(all_uids)
    precision5, recall5, MAP5, ndcg5= [], [], [], []
    precision10, recall10, MAP10, ndcg10= [], [], [], []
    precision20, recall20, MAP20, ndcg20= [], [], [], []
    precision50, recall50, MAP50, ndcg50= [], [], [], []
    print("Start predicting...")
    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            JPF_scores = [JPF.predict(uid, lid, True, POI_cate_matrix, 4)
                              if (uid, lid) not in training_tuples else -1
                              for lid in all_lids]
            CFM_scores = [CFM.predict(uid, lid)
                              if (uid, lid) not in training_tuples else -1
                              for lid in all_lids]

            GKM_scores = [GKM.predict(uid, lid, user_poi_matrix, poi_coos, home_coos)
                              if (uid, lid) not in training_tuples else -1
                              for lid in all_lids]

            JPF_scores = np.array(JPF_scores)
            CFM_scores = np.array(CFM_scores)
            GKM_scores = np.array(GKM_scores)
            overall_scores = CFM_scores * JPF_scores * GKM_scores

            predicted = list(reversed(overall_scores.argsort()))[:top_n]
            actual = ground_truth[uid]

            precision5.append(precisionk(actual, predicted[:5]))
            recall5.append(recallk(actual, predicted[:5]))
            MAP5.append(mapk(actual, predicted[:5], 5))
            ndcg5.append(ndcgk(actual, predicted[:5], 5))

            precision10.append(precisionk(actual, predicted[:10]))
            recall10.append(recallk(actual, predicted[:10]))
            MAP10.append(mapk(actual, predicted[:10], 10))
            ndcg10.append(ndcgk(actual, predicted[:10], 10))

            precision20.append(precisionk(actual, predicted[:20]))
            recall20.append(recallk(actual, predicted[:20]))
            MAP20.append(mapk(actual, predicted[:20], 20))
            ndcg20.append(ndcgk(actual, predicted[:20], 20))

            precision50.append(precisionk(actual, predicted[:50]))
            recall50.append(recallk(actual, predicted[:50]))
            MAP50.append(mapk(actual, predicted[:50], 50))
            ndcg50.append(ndcgk(actual, predicted[:50],50))
            
            result_p.write('\t'.join([str(cnt), str(uid), str(np.mean(precision5)), str(
                np.mean(precision10)), str(np.mean(precision20)), str(np.mean(precision50))]) + '\n')
            result_r.write('\t'.join([str(cnt), str(uid), str(np.mean(recall5)), str(
                np.mean(recall10)), str(np.mean(recall20)), str(np.mean(recall50))]) + '\n')
            result_m.write('\t'.join([str(cnt), str(uid), str(np.mean(MAP5)), str(
                np.mean(MAP10)), str(np.mean(MAP20)), str(np.mean(MAP50))]) + '\n')
            result_n.write('\t'.join([str(cnt), str(uid), str(np.mean(ndcg5)), str(
                np.mean(ndcg10)), str(np.mean(ndcg20)), str(np.mean(ndcg50))]) + '\n')

            print(cnt, uid, "pre@5:", np.mean(precision5), "rec@5:", np.mean(recall5), "map@5:", np.mean(MAP5), "ndcg@5:", np.mean(ndcg5))
            print(cnt, uid, "pre@10:", np.mean(precision10), "rec@10:", np.mean(recall10), "map@10:", np.mean(MAP10), "ndcg@10:", np.mean(ndcg10))
            print(cnt, uid, "pre@20:", np.mean(precision20), "rec@20:", np.mean(recall20), "map@20:", np.mean(MAP20), "ndcg@20:", np.mean(ndcg20))
            print(cnt, uid, "pre@50:", np.mean(precision50), "rec@50:", np.mean(recall50), "map@50:", np.mean(MAP50), "ndcg@50:", np.mean(ndcg50))

    print("Task Finished!")

if __name__ == '__main__':
    data_dir = "../datasets/Yelp/"
    size_file = data_dir + "Yelp_data_size.txt"
    check_in_file = data_dir + "Yelp_checkins.txt"
    train_file = data_dir + "Yelp_train.txt"
    tune_file = data_dir + "Yelp_tune.txt"
    test_file = data_dir + "Yelp_test.txt"
    social_file = data_dir + "Yelp_social_relations.txt"
    category_file = data_dir + "Yelp_user_category.txt"
    poi_file = data_dir + "Yelp_poi_coos.txt"
    user_social_friends_file = data_dir + "Yelp_social_friends.txt"
    user_neighbor_friends_file = data_dir + "Yelp_neighbor_friends.txt"
    user_vector_file = data_dir + "Yelp_user_vector.txt"
    user_features_file = data_dir + "Yelp_user_features.txt"
    user_home_file = data_dir + "Yelp_user_home.txt"
    POI_category_file = data_dir + "Yelp_poi_categories.txt"

    num_users, num_pois, num_categories = open(size_file, 'r').readlines()[0].strip('\n').split()
    num_users, num_pois, num_categories = int(num_users), int(num_pois), int(num_categories)

    top_n = 50

    JPF = JointPoissonFactorModel(K=50, alpha=40.0, beta=0.2)
    CFM_SF = CFMSFModel(eta=0.5)
    CFM_NF = CFMNFModel()
    CFM = CollectiveFriendsModel(lamb = 0.05)
    GKM = GaussianKernelModel()

    main()

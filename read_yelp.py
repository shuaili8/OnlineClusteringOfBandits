import scipy.io as sio
import scipy.sparse as spp
import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
import json

def extract_rows(top_k, sparse_matrix):
    business_review_count=sparse_matrix.getnnz(axis=1)
    business_count=business_review_count.shape[0]
    top_k_index = np.argsort(business_review_count)[business_count-1: business_count -1 -top_k: -1]
    # top_k_index = np.random.choice(business_count, top_k, replace=False)
    matrix=spp.vstack([sparse_matrix.getrow(i) for i in top_k_index])
    return matrix

def extract_cols(top_k, sparse_matrix):
    user_review_count=sparse_matrix.getnnz(axis=0)
    user_count=user_review_count.shape[0]

    top_k_index=np.argsort(user_review_count)[user_count-1: user_count-1-top_k:-1]
    # top_k_index=np.random.choice(user_count, top_k, replace=False)
    matrix=spp.hstack([sparse_matrix.getcol(i) for i in top_k_index])
    return matrix

def load_sparse_matrix(file_name):
    data_list = []
    row_indics_list = []
    col_indics_list = []

    user_dict = {}
    business_dict = {}

    rf = open(file_name)

    l = rf.readline()
    count = 0
    for line in rf:
        dicts = json.loads(line)
        row_index = 0
        col_index = 0
        user_id = dicts["user_id"]
        business_id = dicts["business_id"]
        rating = dicts["stars"]

        if not user_id in user_dict: #.has_key(user_id):
            user_dict[user_id] = len(user_dict)
        row_index = user_dict[user_id]

        if not business_id in business_dict: #.has_key(business_id):
            business_dict[business_id] = len(business_dict)
        col_index = business_dict[business_id]

        #data_list.append(float(rating))
        data_list.append(1)
        row_indics_list.append(row_index)
        col_indics_list.append(col_index)

    data = np.array(data_list)
    rows = np.array(row_indics_list)
    cols = np.array(col_indics_list)
    print(len(data), len(rows), len(cols))
    print(len(user_dict), len(business_dict))
    s_m = spp.csr_matrix((data, (rows, cols)))
    return s_m


def get_reduced_concrete_matrix(user_num, business_num):
    s_m = load_sparse_matrix("review.json")
    row_reduced_matrix=extract_rows(user_num*3, s_m)
    reduced_matrix=extract_cols(business_num, row_reduced_matrix)
    reduced_matrix = extract_rows(user_num, reduced_matrix)
    return reduced_matrix.toarray()

m = get_reduced_concrete_matrix(1000, 1000)
print(m.shape)
np.save("yelp_1000user_1000item.npy", m)

import pandas as pd
import numpy as np
import random
# df = pd.read_csv("results/gqa/trials/results_15.csv") # refine 1 imcomplete code
# logfile = "nohup.out.refine"
df = pd.read_csv("results/gqa/trials/results_16.csv") # baseline 
logfile = "nohup.out.baseline"
# df = pd.read_csv("results/gqa/trials/results_22.csv") # refine 4 test case gen
# logfile = "nohup.out"
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))

from transformers import pipeline

extractor = pipeline(model="google-bert/bert-base-uncased", task="feature-extraction", device="cuda:6")
# result = extractor("This is a simple test.", return_tensors=True)
# result.shape  # This is a tensor of shape [1, sequence_length, hidden_dimension] representing the input string.

import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4, seed=42):
    """
    X: 数据矩阵 (n_samples, n_features)
    k: 聚类个数
    max_iters: 最大迭代次数
    tol: 收敛阈值
    seed: 随机种子
    """
    np.random.seed(seed)
    
    n_samples, n_features = X.shape

    # 1. 随机初始化 k 个聚类中心
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices]

    for _ in range(max_iters):
        # 2. 计算每个样本到各个中心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # 3. 分配样本到最近的中心
        labels = np.argmin(distances, axis=1)

        # 4. 计算新的中心
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # 5. 判断收敛（中心移动是否小于 tol）
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return centroids, labels


import json
datas = {}
with open('result_soundness.json', 'r', encoding='utf8') as f:
    datas = json.load(f)

# filter yes or no


features = []
trace_backs = []
cnt = 0
cnt_yn = 0
for key, data in datas.items():
    if data["soundness"] != 1.0:
        continue
    # print(data["test_code"])
     
    if """assert result in ['yes', 'no']""" in data["test_code"]:
        # feature["soundness"] = 0.0
        # continue
        # print("found")
        cnt_yn += 1
        continue
   
    feature = extractor(data["query"] + '_' + data["answer"] + '_' + data["test_code"], return_tensors=True)[0]
    # print(feature.shape)
    # feature = extractor(data["test_code"], return_tensors=True)[0]
    # print(feature.shape)
    feature = feature.mean(axis=0)
    # print(feature.shape)
    features.append(feature)
    trace_back = (key, cnt)
    trace_backs.append(trace_back)
    cnt += 1
    # print(data["query"])
X = np.stack(features)
print(cnt)
# print(X.shape)
for k in [10]:
    centroids, labels = kmeans(X, k=k)
    print(f"K={k}")
    print("Centroids shape:", centroids.shape)
    print("labels:", labels)

    cluster_to_indices = {}

    # 將每個 label 的 index 收集起來
    for idx, lab in enumerate(labels):
        cluster_to_indices.setdefault(lab, []).append(idx)

    # 從每個 cluster 隨機取一個樣本 index
    print("Random sample from each cluster:")
    for lab, idx_list in cluster_to_indices.items():
        choice = random.choice(idx_list)
        # print(f"Cluster {lab}: index {choice}, key: {trace_backs[choice][0]}, query: {datas[trace_backs[choice][0]]['query']}, answer: {datas[trace_backs[choice][0]]['answer']}, test_code: {datas[trace_backs[choice][0]]['test_code']}")
        print(f"""
Query: {datas[trace_backs[choice][0]]['query']}

{datas[trace_backs[choice][0]]['test_code']}
###



""")
 


import numpy as np
import pandas as pd 

import json

from metrics import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str)
args = parser.parse_args()

test_df = pd.read_json("data/ACL_200/test.jsonl", orient='records', lines=True)
papers_df = pd.read_json("data/ACL_200/papers.jsonl", orient='records', lines=True)
pred_df = pd.read_csv(args.pred_path, header=None)

# print(test_df)
# print(papers_df)
# print(pred_df)

n = len(test_df)
assert(n == len(pred_df))

test_dataset = json.load(open("data/acl/test.json"))

# load BM25 candidates retrieved in [Medi´c and ˇSnajder, 2020]
BM25_candidates = []
for i in range(n):
    tmp = []
    for j in range(2000):
        tmp.append(test_dataset[i * 2000 + j]["paper_id"])
    BM25_candidates.append(tmp)

# print("BM25 candidates samples.")
# print(BM25_candidates[0][:10])



print("all candidates")

rc = []
rr = []

for index, row in enumerate(test_df.itertuples()):
    true_label = row[9]

    predictions = []
    for i in range(10):
        label_id = pred_df.at[index, i]
        paper_id = papers_df.at[papers_df.index[label_id], 'id']
        predictions.append(paper_id)

    arr = [1 if i == true_label else 0 for i in predictions]
    rc.append(recall(arr))
    rr.append(reciprocal_rank(arr))

print(f"Recall@10: {round(np.mean(rc), 3)}")
print(f"MRR      : {round(np.mean(rr), 3)}")



print("BM25 candidates")

rc = []
rr = []

for index, row in enumerate(test_df.itertuples()):
    true_label = row[9]

    predictions = []
    for i in range(2000):
        label_id = pred_df.at[index, i]
        paper_id = papers_df.at[papers_df.index[label_id], 'id']
        if paper_id in BM25_candidates[index]:
            predictions.append(paper_id)
        if len(predictions) == 10:
            break

    if len(predictions) < 10:
        print("prediction topk is not enough")
        for i in range(10 - len(predictions)):
            predictions.append(None)

    arr = [1 if i == true_label else 0 for i in predictions]
    rc.append(recall(arr))
    rr.append(reciprocal_rank(arr))

print(f"Recall@10: {round(np.mean(rc), 3)}")
print(f"MRR      : {round(np.mean(rr), 3)}")
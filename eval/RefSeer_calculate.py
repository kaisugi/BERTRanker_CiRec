import numpy as np
import pandas as pd 
from tqdm import tqdm

import json

from metrics import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str)
args = parser.parse_args()


test_df = pd.read_json("data/RefSeer/test.jsonl", orient='records', lines=True)
print("loaded test.jsonl")
papers_df = pd.read_json("data/RefSeer/papers.jsonl", orient='records', lines=True)
print("loaded papers.jsonl")
pred_file = open(args.pred_path)

# print(test_df)
# print(papers_df)
# print(pred_df)

n = len(test_df)
# assert(n == len(pred_df))

papers_arr = papers_df["id"].values.tolist()
print("get papers_arr")

test_dataset = pd.read_csv(open("data/refseer/test.csv"), header=None)
print("loaded test.csv")

# load BM25 candidates retrieved in [Medi´c and ˇSnajder, 2020]
BM25_candidates = []
for i in tqdm(range(n)):
    tmp = []
    for j in range(2048):
        tmp.append(test_dataset.at[i, j+1])
    BM25_candidates.append(tmp)

# print("BM25 candidates samples.")
# print(BM25_candidates[0][:10])



print("all candidates")

rc = []
rr = []

for index, row in enumerate(tqdm(test_df.itertuples())):
    # localで7, globalで9
    true_label = row[9]

    pred_line = pred_file.readline()
    pred_arr = pred_line.replace("\n", "").split(",")
    pred_arr = [int(i) for i in pred_arr]

    predictions = []
    for i in range(10):
        label_id = pred_arr[i]
        paper_id = papers_arr[label_id]
        predictions.append(paper_id)

    arr = [1 if i == true_label else 0 for i in predictions]
    rc.append(recall(arr))
    rr.append(reciprocal_rank(arr))

print(f"Recall@10: {round(np.mean(rc), 3)}")
print(f"MRR      : {round(np.mean(rr), 3)}")
pred_file.close()



pred_file = open(args.pred_path)
print("BM25 candidates")

rc = []
rr = []

for index, row in enumerate(tqdm(test_df.itertuples())):
    # localで7, globalで9
    true_label = row[9]

    pred_line = pred_file.readline()
    pred_arr = pred_line.replace("\n", "").split(",")
    pred_arr = [int(i) for i in pred_arr]

    predictions = []
    
    for i in range(60000):
        label_id = pred_arr[i]
        paper_id = papers_arr[label_id]
        if paper_id in BM25_candidates[index]:
            predictions.append(paper_id)
        if len(predictions) == 10:
            break

    if len(predictions) < 10:
        print(index)
        print("prediction topk is not enough")
        for i in range(10 - len(predictions)):
            predictions.append(None)

    arr = [1 if i == true_label else 0 for i in predictions]
    rc.append(recall(arr))
    rr.append(reciprocal_rank(arr))

print(f"Recall@10: {round(np.mean(rc), 3)}")
print(f"MRR      : {round(np.mean(rr), 3)}")
pred_file.close()
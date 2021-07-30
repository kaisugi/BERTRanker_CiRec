import json
import logging
import torch
from tqdm import tqdm

import time

import utils


class Stats():
    def __init__(self, top_k=1000):
        self.cnt = 0
        self.hits = []
        self.top_k = top_k
        self.rank = [1, 4, 8, 16, 32, 64, 100, 128, 256, 512]
        self.LEN = len(self.rank) 
        for i in range(self.LEN):
            self.hits.append(0)

    def add(self, idx):
        self.cnt += 1
        if idx == -1:
            return
        for i in range(self.LEN):
            if idx < self.rank[i]:
                self.hits[i] += 1

    def extend(self, stats):
        self.cnt += stats.cnt
        for i in range(self.LEN):
            self.hits[i] += stats.hits[i]

    def output(self):
        output_json = "Total: %d examples." % self.cnt
        for i in range(self.LEN):
            if self.top_k < self.rank[i]:
                break
            output_json += " r@%d: %.4f" % (self.rank[i], self.hits[i] / float(self.cnt))
        return output_json


def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool,
    cand_encode_list,
    silent,
    global_info,
    logger,
    output_path,
    top_k=10
):
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_context = []
    nn_candidates = []
    nn_labels = []
    stats = {}

    world_size = 1
    candidate_pool = [candidate_pool]
    cand_encode_list = [cand_encode_list]

    for i in range(world_size):
        stats[i] = Stats(top_k)


    f = open(f"{output_path}/predictions.txt", "w")
    

    oid = 0

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)

        if global_info:
            global_info_input, context_input, _, label_ids = batch
            src = 0
            scores = reranker.global_score_candidate(
                global_info_input,
                context_input, 
                None, 
                cand_encs=cand_encode_list[src].to(device)
            )
        else:
            context_input, _, label_ids = batch
            src = 0
            scores = reranker.score_candidate(
                context_input, 
                None, 
                cand_encs=cand_encode_list[src].to(device)
            )
        
        values, indicies = scores.topk(top_k)

        for i in range(context_input.size(0)):
            oid += 1
            inds = indicies[i]

            inds_list = inds.tolist()
            inds_list = [str(i) for i in inds_list]
            f.write(f"{', '.join(inds_list)}\n")

#            pointer = -1
#            for j in range(top_k):
#                if inds[j].item() == label_ids[i].item():
#                    pointer = j
#                    break
#            stats[src].add(pointer)
#
#            if pointer == -1:
#                continue
#
#            # add examples in new_data
#            cur_candidates = candidate_pool[src][inds]
#            nn_context.append(context_input[i].cpu().tolist())
#            nn_candidates.append(cur_candidates.cpu().tolist())
#            nn_labels.append(pointer)

    f.close()

#    res = Stats(top_k)
#    for src in range(world_size):
#        if stats[src].cnt == 0:
#            continue
#
#        output = stats[src].output()
#        # logger.info(output)
#        res.extend(stats[src])
#
#    # logger.info(res.output())
#
#    nn_context = torch.LongTensor(nn_context)
#    nn_candidates = torch.LongTensor(nn_candidates)
#    nn_labels = torch.LongTensor(nn_labels)
#    nn_data = {
#        'context_vecs': nn_context,
#        'candidate_vecs': nn_candidates,
#        'labels': nn_labels,
#    }
#    
#    return nn_data


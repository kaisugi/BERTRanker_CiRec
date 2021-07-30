import logging
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from common.params import CITATION_TAG, ENT_TITLE_TAG


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    context_key="context",
    citation_token=CITATION_TAG
):
    citation_tokens = [citation_token]

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(citation_tokens)) // 2 - 1
    right_quota = max_seq_length - len(citation_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + citation_tokens + context_right[:right_quota]
    )

    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
    candidate_desc, # 本体 (description)
    tokenizer, 
    title_only,
    max_seq_length, 
    candidate_title=None, # タイトル
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    if title_only:
        cand_tokens = title_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_citation_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    title_only,
    global_info,
    context_key="context",
    citing_label_key="citing",
    citing_title_key="citing_title",
    label_key="label",
    title_key='label_title',
    citation_token=CITATION_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            context_key,
            citation_token
        )

        label = sample[label_key]
        title = sample.get(title_key, None)
        label_tokens = get_candidate_representation(
            label, tokenizer, title_only, max_cand_length, title,
        )
        label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
        }

        if global_info:
            citing_label = sample[citing_label_key]
            citing_title = sample[citing_title_key]
            citing_label_tokens = get_candidate_representation(
                citing_label, tokenizer, title_only, max_cand_length, citing_title,
            )
            record["global_info"] = citing_label_tokens

        processed_samples.append(record)

    if logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            if global_info:
                logger.info("GlobalInfo tokens : " + " ".join(sample["global_info"]["tokens"]))
                logger.info(
                    "GlobalInfo ids : " + " ".join([str(v) for v in sample["global_info"]["ids"]])
                )

            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )

    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx)

    if global_info:
        global_info_vecs = torch.tensor(
            select_field(processed_samples, "global_info", "ids"), dtype=torch.long,
        )
        data["global_info_vecs"] = global_info_vecs
        tensor_data = TensorDataset(global_info_vecs, context_vecs, cand_vecs, label_idx)

    return data, tensor_data

# BERTRanker_CiRec

Codes for "Context-aware Citation Recommendation Based on BERT-based Bi-Ranker" (SciNLP 2021)  

If you have any questions or comments, feel free to open an issue.

## Setup

```
pyenv local 3.7.0
poetry install
wget -P bertmodels https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar
tar xvf bertmodels/scibert_scivocab_uncased.tar -C bertmodels
poetry shell
```

## Data

We run experiments on three different datasets, ACL-600, ACL-200, and RefSeer. These datasets are the same as those used in [DualLCR](https://github.com/zoranmedic/DualLCR), but we have changed the file format from `.json` to `.jsonl` so that we can use them more smoothly in our models. The modified datasets can be downloaded from [here](https://drive.google.com/drive/folders/1Lcb9SbjWjtizrLdoRU4GKpXiwDvz7jm-?usp=sharing). Please set them in `/data` directory.

We compare the proposed models in two cases: one where candidate documents are extracted in advance using BM25, and one where they are not. The BM25 candidates are the same one that was used in [DualLCR](https://github.com/zoranmedic/DualLCR). So, if you would like to reproduce the results of the inferences, you need to download `acl/test.json` and `refseer/test.csv` from the links in DualLCR repository (because these test sets include BM25 candidates). They have to be placed under `/data/acl` and `/data/refseer` respectively.

## Training

**You can skip this process by using the trained model listed below.**  

**models/ACL_600**: https://huggingface.co/kaisugi/BERTRanker_CiRec_ACL600  
**models/ACL_600_global**: https://huggingface.co/kaisugi/BERTRanker_CiRec_ACL600_global  
**models/ACL_200**: https://huggingface.co/kaisugi/BERTRanker_CiRec_ACL200  
**models/ACL_200_global**: https://huggingface.co/kaisugi/BERTRanker_CiRec_ACL200_global  
**models/RefSeer**: https://huggingface.co/kaisugi/BERTRanker_CiRec_RefSeer  
**models/RefSeer_global**: https://huggingface.co/kaisugi/BERTRanker_CiRec_RefSeer_global


If you would like to evaluate a global model that considers the title and abstract of the citing paper, add the parameter `--global_info`.

### ACL_600

```
python train_biencoder.py \
    --data_path data/ACL_600 \
    --max_context_length 256 \
    --max_cand_length 256 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --shuffle True \
    --output_path output/models/ACL_600
```

### ACL_200

```
python train_biencoder.py \
    --data_path data/ACL_200 \
    --max_context_length 128 \
    --max_cand_length 256 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --shuffle True \
    --output_path output/models/ACL_200
```

### RefSeer

```
python train_biencoder.py \
    --data_path data/RefSeer \
    --max_context_length 128 \
    --max_cand_length 256 \
    --train_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --shuffle True \
    --eval_interval 8000 \
    --output_path output/models/RefSeer
```

## Evaluation

If you would like to evaluate a global model that considers the title and abstract of the citing paper, add the parameter `--global_info`.

### ACL_600

```
python test_biencoder.py \
    --data_path data/ACL_600 \
    --entity_dict_path data/ACL_600/papers.jsonl \
    --max_context_length 256 \
    --max_cand_length 256 \
    --path_to_model output/models/ACL_600/pytorch_model.bin \
    --output_path output/predictions/ACL_600
```

```
python eval/ACL600_calculate.py --pred_path output/predictions/ACL_600/predictions.txt
```

### ACL_200

```
python test_biencoder.py \
    --data_path data/ACL_200 \
    --entity_dict_path data/ACL_200/papers.jsonl \
    --max_context_length 128 \
    --max_cand_length 256 \
    --path_to_model output/models/ACL_200/pytorch_model.bin \
    --output_path output/predictions/ACL_200
```

```
python eval/ACL200_calculate.py --pred_path output/predictions/ACL_200/predictions.txt
```

### RefSeer

Since the data size is very large, we recommend caching the embedding of candidate papers using the `--cand_encode_path` parameter.

```
python test_biencoder.py \
    --data_path data/RefSeer \
    --entity_dict_path data/RefSeer/papers.jsonl \
    --max_context_length 128 \
    --max_cand_length 256 \
    --path_to_model output/models/RefSeer/pytorch_model.bin \
    --cand_encode_path candidate_encoding/RefSeer \
    --top_k 60000 \
    --output_path output/predictions/RefSeer
```

```
python eval/RefSeer_calculate.py --pred_path output/predictions/RefSeer/predictions.txt
```

## Citation

```
@inproceedings{kaito-2021-context,
    title = "Context-aware {C}itation {R}ecommendation {B}ased on {BERT}-based {B}i-{R}anker",
    author = "Sugimoto, Kaito  and
      Aizawa, Akiko",
    booktitle = "2nd Workshop on Natural Language Processing for Scientific Text at AKBC 2021",
    month = oct,
    year = "2021",
    address = "Online"
}
```

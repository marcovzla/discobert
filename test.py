#!/usr/bin/env python

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from model import DiscoBertModel
from model_glove import DiscoBertModelGlove
from rst import load_annotations, iter_spans_only, iter_nuclearity_spans, iter_labeled_spans, iter_labeled_spans_with_nuclearity
from utils import prf1
import config
import engine
import random
import os
import sys
import shutil
from datetime import date
import time
from utils import make_word2index

def optimizer_parameters(model):
    no_decay = ['bias', 'LayerNorm']
    named_params = list(model.named_parameters())
    return [
        {'params': [p for n,p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n,p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

def eval_trees(pred_trees, gold_trees, view_fn):
    all_pred_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in pred_trees]
    all_gold_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in gold_trees]
    scores = [prf1(pred, gold) for pred, gold in zip(all_pred_spans, all_gold_spans)]
    scores = np.array(scores).mean(axis=0).tolist()
    return scores

def main(path_to_model, test_ds, random_seed):
    #set random seed here because the vocab will be built based on the train set
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    device = torch.device('cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu')
    if config.ENCODING == "glove":
        
        # load data and split in train and validation sets
        train_ds, valid_ds = train_test_split(list(load_annotations(config.TRAIN_PATH)), test_size=config.TEST_SIZE)

        word2index = make_word2index(train_ds)   
        model = DiscoBertModelGlove(word2index).load(path_to_model, word2index)
        
    else:
        model = DiscoBertModel.load(path_to_model)
    model.to(device)
    
    pred_trees, gold_trees = engine.eval_fn(test_ds, model, device)
    p, r, f1_s = eval_trees(pred_trees, gold_trees, iter_spans_only)
    # print(f'S (span only)   P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
    print(f'S (span only)   F1:{f1_s:.2%}')
        
    p, r, f1_n = eval_trees(pred_trees, gold_trees, iter_nuclearity_spans)
    # print(f'N (span + dir)  P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
    print(f'N (span + dir)  F1:{f1_n:.2%}')
       

    p, r, f1_r = eval_trees(pred_trees, gold_trees, iter_labeled_spans)
    # print(f'R (span + label)        P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
    print(f'R (span + label)        F1:{f1_r:.2%}')
       
    p, r, f1 = eval_trees(pred_trees, gold_trees, iter_labeled_spans_with_nuclearity)
    # print(f'F (full)        P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
    print(f'F (full)        F1:{f1:.2%}')

    return f1_s, f1_n, f1_r, f1
    
    
if __name__ == '__main__':

    test_ds = list(load_annotations(config.VALID_PATH))
    experiment_dir_path = config.OUTPUT_DIR/config.EXPERIMENT_DESCRIPTION

    with open(os.path.join(experiment_dir_path, "eval_log"), "w") as f:
        sys.stdout = f
        random_seeds = config.RANDOM_SEEDS

        span_scores = np.zeros(len(random_seeds))
        nuclearity_scores = np.zeros(len(random_seeds)) # span + direction
        relations_scores = np.zeros(len(random_seeds)) # span + relation label
        full_scores = np.zeros(len(random_seeds)) # span + direction + relation label

        for i in range(len(random_seeds)):
            rs = random_seeds[i]
            path_to_model = os.path.join(str(experiment_dir_path/'rs') + str(rs), config.MODEL_FILENAME)
            print("model path: ", path_to_model)
            rs_results = main(path_to_model, test_ds, rs)
            span_scores[i] = rs_results[0]
            nuclearity_scores[i] = rs_results[1]
            relations_scores[i] = rs_results[2]
            full_scores[i] = rs_results[3]

        print("\n========================================================")
        print(f"Mean scores from {len(random_seeds)} runs with different random seeds:")
        print("--------------------------------------------------------")
        print("F1 (span):\t", np.around(np.mean(span_scores), decimals=3), "±", np.around(np.std(span_scores), decimals=3))
        print("F1 (span + dir):\t", np.around(np.mean(nuclearity_scores), decimals=3), "±", np.around(np.std(nuclearity_scores), decimals=3))
        print("F1 (span + rel):\t", np.around(np.mean(relations_scores), decimals=3), "±", np.around(np.std(relations_scores), decimals=3))
        print("F1 (full):\t", np.around(np.mean(full_scores), decimals=3), "±", np.around(np.std(full_scores), decimals=3))
 



        
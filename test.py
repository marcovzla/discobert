#!/usr/bin/env python

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from model import DiscoBertModel
from rst import load_annotations, iter_spans_only, iter_nuclearity_spans, iter_labeled_spans, iter_labeled_spans_with_nuclearity
from utils import prf1, tpfpfn, calc_prf_from_tpfpfn
import config
import engine
import random
import os
import sys
import shutil
from datetime import date
import time


def optimizer_parameters(model):
    no_decay = ['bias', 'LayerNorm']
    named_params = list(model.named_parameters())
    return [
        {'params': [p for n,p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n,p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

# def eval_trees(pred_trees, gold_trees, view_fn):
#     all_pred_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in pred_trees]
#     all_gold_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in gold_trees]
#     scores = [prf1(pred, gold) for pred, gold in zip(all_pred_spans, all_gold_spans)]
#     print(scores)
#     scores = np.array(scores).mean(axis=0).tolist()
#     print(scores)
#     return scores

def eval_trees(pred_trees, gold_trees, view_fn):
    all_pred_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in pred_trees]
    all_gold_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in gold_trees]
    tpfpfns = [tpfpfn(pred, gold) for pred, gold in zip(all_pred_spans, all_gold_spans)]
    # print(tpfpfns)
    tp, fp, fn = np.array(tpfpfns).sum(axis=0)
    # print(tp, fp, fn)
    scores = calc_prf_from_tpfpfn(tp, fp, fn)
    # scores = np.array(scores).mean(axis=0).tolist()
    # print(scores)
    return scores

# def eval_trees(pred_trees, gold_trees, view_fn):
#     all_pred_spans = [f'{x}' for t in pred_trees for x in view_fn(t.get_nonterminals())]
#     all_gold_spans = [f'{x}' for t in gold_trees for x in view_fn(t.get_nonterminals())]
#     scores = prf1(all_pred_spans, all_gold_spans)
#     return scores

def main(path_to_model, test_ds):
    
    if config.RERUN_DEV_EVAL == True:
        train_ds, valid_ds = train_test_split(list(load_annotations(config.TRAIN_PATH)), test_size=config.TEST_SIZE)
    device = torch.device('cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu')
    model = DiscoBertModel.load(path_to_model)
    model.to(device)
    
    if config.RERUN_DEV_EVAL == True:
        pred_trees, gold_trees = engine.eval_fn(valid_ds, model, device)
    else:
        pred_trees, gold_trees = engine.eval_fn(test_ds, model, device)

    if config.PRINT_TREES == False:
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
    else:
        return None
    
if __name__ == '__main__':

    test_ds = list(load_annotations(config.VALID_PATH))
    experiment_dir_path = config.OUTPUT_DIR/config.EXPERIMENT_DESCRIPTION
    random_seeds = config.RANDOM_SEEDS
    if config.PRINT_TREES == False:
        if config.RERUN_DEV_EVAL == True:
            log_name = "eval_log-dev-new-eval-with-tpfpfn"
        else:
            log_name = "eval_log-new-eval-with-tpfpfn"
        with open(os.path.join(experiment_dir_path, log_name), "w") as f:
            sys.stdout = f
            

            span_scores = np.zeros(len(random_seeds))
            nuclearity_scores = np.zeros(len(random_seeds)) # span + direction
            relations_scores = np.zeros(len(random_seeds)) # span + relation label
            full_scores = np.zeros(len(random_seeds)) # span + direction + relation label

            for i in range(len(random_seeds)):
                r_seed = random_seeds[i]
                random.seed(r_seed)
                torch.manual_seed(r_seed)
                torch.cuda.manual_seed(r_seed)
                np.random.seed(r_seed)
                path_to_model = os.path.join(str(experiment_dir_path/'rs') + str(r_seed), config.MODEL_FILENAME)
                print("model path: ", path_to_model)
                rs_results = main(path_to_model, test_ds)
                span_scores[i] = rs_results[0]
                nuclearity_scores[i] = rs_results[1]
                relations_scores[i] = rs_results[2]
                full_scores[i] = rs_results[3]

            span_score = np.around(np.mean(span_scores), decimals=3)
            span_score_sd = np.around(np.std(span_scores), decimals=3)
            nuc_score = np.around(np.mean(nuclearity_scores), decimals=3)
            nuc_score_sd = np.around(np.std(nuclearity_scores), decimals=3)
            rel_score = np.around(np.mean(relations_scores), decimals=3)
            rel_score_sd = np.around(np.std(relations_scores), decimals=3)
            full_score = np.around(np.mean(full_scores), decimals=3)
            full_score_sd = np.around(np.std(full_scores), decimals=3)
            print("\n========================================================")
            print(f"Mean scores from {len(random_seeds)} runs with different random seeds:")
            print("--------------------------------------------------------")
            print("F1 (span):\t", span_score, "±", span_score_sd)
            print("F1 (span + dir):\t", nuc_score , "±", nuc_score_sd)
            print("F1 (span + rel):\t", rel_score, "±", rel_score_sd)
            print("F1 (full):\t", full_score , "±", full_score_sd)
            textpm_string = "\\\\textpm".replace("\\\\", "\\")
            print("latex printout: ", f" & {span_score} {textpm_string} {span_score_sd} &  {nuc_score} {textpm_string} {nuc_score_sd} &  {rel_score} {textpm_string} {rel_score_sd} & {full_score} {textpm_string} {full_score_sd} \\\\")

    else:
        for i in range(len(random_seeds)):
            r_seed = random_seeds[i]
            random.seed(r_seed)
            torch.manual_seed(r_seed)
            torch.cuda.manual_seed(r_seed)
            np.random.seed(r_seed)
            path_to_model = os.path.join(str(experiment_dir_path/'rs') + str(r_seed), config.MODEL_FILENAME)
            # print("model path: ", path_to_model)
            
            log_name = "trees_" + str(r_seed) + ".txt"
            with open(os.path.join(experiment_dir_path, log_name), "w") as f:
                sys.stdout = f
                rs_results = main(path_to_model, test_ds)
            
                


        
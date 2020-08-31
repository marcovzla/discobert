#!/usr/bin/env python

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from segmenter_model import SegmentationModel
from rst import load_annotations, iter_spans_only, iter_nuclearity_spans, iter_labeled_spans, iter_labeled_spans_with_nuclearity
from utils import prf1
import config
import segmenter_engine
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

def eval_boundaries(predictions, gold_boundaries):
    # print("predictions: ", predictions)
    # print("golds: ", gold_boundaries)
    # print("preds len: ", len(predictions))
    # print("golds len: ", len(gold_boundaries))
    # following braud (2017), to eval the segmenter, just calculated p,r,f1 for boundaries
    correct_bs = 0 #true pos
    wrong_bs = 0 #false pos
    missed_bs = 0 # false neg

    for i in range(len(predictions)):
        # print("gold: ", gold_boundaries[i], " pred: ", predictions[i])
        if predictions[i] == "B" and gold_boundaries[i] == "B":
            # print("CORRECT")
            correct_bs += 1
        
        elif predictions[i] == "B" and gold_boundaries[i] != "B":
            wrong_bs +=1 
            # print("FALSE POS")
        elif gold_boundaries[i] == "B" and predictions[i] != "B":
            missed_bs += 1
            # print("FALSE NEG")
    print("cor bs: ", correct_bs)
    print("wrong bs: ", wrong_bs)
    print("missed bs: ", missed_bs)
    precision = float(correct_bs)/(correct_bs + wrong_bs)
    recall =  float(correct_bs)/(correct_bs + missed_bs) 
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def main(path_to_model, test_ds):
    
    device = torch.device('cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu')
    model = SegmentationModel.load(path_to_model)
    model.to(device)
    
    pred_trees, gold_trees = segmenter_engine.eval_fn(test_ds, model, device)
    p, r, f1 = eval_boundaries(pred_trees, gold_trees)
    print(f'boundaries   P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        
    

    return p, r, f1
    
    
if __name__ == '__main__':

    test_ds = list(load_annotations(config.VALID_PATH))
    experiment_dir_path = config.SEGMENTER_OUTPUT_DIR/config.EXPERIMENT_DESCRIPTION

    with open(os.path.join(experiment_dir_path, "eval_log"), "w") as f:
        sys.stdout = f
        random_seeds = config.RANDOM_SEEDS

        p_scores = np.zeros(len(random_seeds))
        r_scores = np.zeros(len(random_seeds)) 
        f1_scores = np.zeros(len(random_seeds))

        for i in range(len(random_seeds)):
            rs = random_seeds[i]
            path_to_model = os.path.join(str(experiment_dir_path/'rs') + str(rs), config.SEGMENTER_MODEL_FILENAME)
            print("model path: ", path_to_model)
            rs_results = main(path_to_model, test_ds)
            p_scores[i] = rs_results[0]
            r_scores[i] = rs_results[1]
            f1_scores[i] = rs_results[2]
       

        p_score = np.around(np.mean(p_scores), decimals=3)
        p_score_sd = np.around(np.std(p_scores), decimals=3)
        r_score = np.around(np.mean(r_scores), decimals=3)
        r_score_sd = np.around(np.std(r_scores), decimals=3)
        f1_score = np.around(np.mean(f1_scores), decimals=3)
        f1_score_sd = np.around(np.std(f1_scores), decimals=3)
        
        print("\n========================================================")
        print(f"Mean scores from {len(random_seeds)} runs with different random seeds:")
        print("--------------------------------------------------------")
        print("Precision:\t", p_score, "±", p_score_sd)
        print("Recall:\t", r_score , "±", r_score_sd)
        print("F1:\t", f1_score, "±", f1_score_sd)
  
        # print("latex pringout: ", f" & {span_score} {textpm_string} {span_score_sd} &  {nuc_score} {textpm_string} {nuc_score_sd} &  {rel_score} {textpm_string} {rel_score_sd} & {full_score} {textpm_string} {full_score_sd} \\\\")



        
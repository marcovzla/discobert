#!/usr/bin/env python

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from model import DiscoBertModel
from rst import load_annotations, iter_spans_only, iter_nuclearity_spans, iter_labeled_spans, iter_labeled_spans_with_nuclearity
from utils import prf1
import config
import engine
import random
import os




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

def main():
    model_dir_path = os.path.join(os.path.join(config.OUTPUT_DIR, "experiment" + str(config.EXPERIMENT_ID)), "rs" + str(r_seed))
    print("model dir path: ", model_dir_path)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    model_path = os.path.join(model_dir_path, config.MODEL_FILENAME)
    print("model path: ", model_path)
    device = torch.device('cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu')
    model = DiscoBertModel()
    model.to(device)

    train_ds, valid_ds = train_test_split(list(load_annotations(config.TRAIN_PATH)))

    num_training_steps = int(len(train_ds) * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters(model), lr=config.LR, eps=1e-8, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    best_f1 = 0
    
    max_f1_S = 0
    max_f1_N = 0
    max_f1_R = 0
    max_f1_F = 0
    max_f1_S_epoch = 1
    max_f1_N_epoch = 1
    max_f1_R_epoch = 1
    max_f1_F_epoch = 1
    for epoch in range(config.EPOCHS):
        if epoch > 0: print()
        print(f'epoch: {epoch+1}/{config.EPOCHS}')
        engine.train_fn(train_ds, model, optimizer, device, scheduler)
        pred_trees, gold_trees = engine.eval_fn(valid_ds, model, device)
        p, r, f1_span = eval_trees(pred_trees, gold_trees, iter_spans_only)
        # print(f'S (span only)   P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        if f1_span > max_f1_S:
            max_f1_S = f1_span
            max_f1_S_epoch = epoch

        print(f'S (span only)   F1:{f1_span:.2%}')
        p, r, f1_n = eval_trees(pred_trees, gold_trees, iter_nuclearity_spans)
        # print(f'N (span + dir)  P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        if f1_n > max_f1_N:
            max_f1_N = f1_n
            max_f1_N_epoch = epoch
        print(f'N (span + dir)  F1:{f1_n:.2%}')

        p, r, f1_r = eval_trees(pred_trees, gold_trees, iter_labeled_spans)
        # print(f'R (span + label)        P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        print(f'R (span + label)        F1:{f1_r:.2%}')
        if f1_r > max_f1_R:
            max_f1_R = f1_r
            max_f1_R_epoch = epoch
        p, r, f1 = eval_trees(pred_trees, gold_trees, iter_labeled_spans_with_nuclearity)
        # print(f'F (full)        P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        print(f'F (full)        F1:{f1:.2%}')
        if f1 > max_f1_F:
            max_f1_F = f1
            max_f1_F_epoch = epoch
        if f1 > best_f1:
            model.save(model_path)
            best_f1 = f1

        print("max f1 span: ", max_f1_S, " ", max_f1_S_epoch)
        print("max f1 span + dir: ", max_f1_N, " ", max_f1_N_epoch)
        print("max f1 span + rel: ", max_f1_R, " ", max_f1_R_epoch)
        print("max f1 span + rel + dir: ", max_f1_F, " ", max_f1_F_epoch)
        return f1_span, f1_n, f1_r, f1
        
        

if __name__ == '__main__':

    #todo: cp config into experiment folder
    #log outputs from both main() (one log file per random seed) and log from this main method with overall results
    #also output here which rs gave best f1
    f1_s_overall = 0
    f1_n_overall = 0
    f1_r_overall = 0
    f1_overall = 0
    random_seeds = config.RANDOM_SEEDS
    for r_seed in random_seeds:
        
        random.seed(r_seed)
        torch.manual_seed(r_seed)
        torch.cuda.manual_seed(r_seed)
        np.random.seed(r_seed)
        rs_results = main()
        f1_s_overall += rs_results[0]
        f1_n_overall += rs_results[1]
        f1_r_overall += rs_results[2]
        f1_overall += rs_results[3]
    print(f1_s_overall/len(random_seeds), "<-")
    print(f1_n_overall/len(random_seeds))
    print(f1_r_overall/len(random_seeds))
    print(f1_overall/len(random_seeds))


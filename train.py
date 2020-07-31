#!/usr/bin/env python

import torch
import torchtext
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from model import DiscoBertModel
from model_glove import DiscoBertModelGlove
from model_glove_2_class import DiscoBertModelGlove2Class
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

def main(experiment_dir_path):

    print("Printing out config settings:")
    print("debug: ", config.DEBUG)
    print("encoding: ", config.ENCODING)
    print("tokenizer: ", config.TOKENIZER)
    # print("model: ", config.MODEL)
    print("use attention", config.USE_ATTENTION)
    print("use relation and dir emb-s: ", config.INCLUDE_RELATION_EMBEDDING, " ", config.INCLUDE_DIRECTION_EMBEDDING)
    print("sort input: ", config.SORT_INPUT)
    print("test size: ", config.TEST_SIZE)

    # load data and split in train and validation sets
    train_ds, valid_ds = train_test_split(list(load_annotations(config.TRAIN_PATH)), test_size=config.TEST_SIZE)


    if config.DEBUG == False:
        model_dir_path = os.path.join(experiment_dir_path, "rs" + str(r_seed))
        # print("model dir path: ", model_dir_path)
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        model_path = os.path.join(model_dir_path, config.MODEL_FILENAME)
        print("path to model: ", model_path)

    
    if config.SORT_INPUT == True:
        # construct new train_ds
        train_ids_by_length = {}
        for item in train_ds:
            train_ids_by_length.setdefault(len(item.edus), []).append(item)

        train_ds = []
        for n in sorted(train_ids_by_length):
            for ann in train_ids_by_length[n]:
                train_ds.append(ann)

    device = torch.device('cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu')
    if config.ENCODING == 'glove':
        word2index = make_word2index(train_ds)   
        model = DiscoBertModelGlove(word2index)
    elif config.ENCODING == 'glove-2-class':
        word2index = make_word2index(train_ds)   
        model = DiscoBertModelGlove2Class(word2index)
    else:
        model = DiscoBertModel()
        
    
    model.to(device)

    num_training_steps = int(len(train_ds) * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters(model), lr=config.LR, eps=1e-8, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # the scores from the model that was best based on Full F1 score and that was saved
    saved_model_f1_s = 0
    saved_model_f1_n = 0
    saved_model_f1_r = 0
    saved_model_f1_f = 0
    
    # max component scores and the associated epochs
    # Note: saved_model_f1_f should be equal to max_f1_F
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
        print("-----------")
        print(f'epoch: {epoch+1}/{config.EPOCHS}')
        print("-----------")
        engine.train_fn(train_ds, model, optimizer, device, scheduler)
        pred_trees, gold_trees = engine.eval_fn(valid_ds, model, device)
        p, r, f1_s = eval_trees(pred_trees, gold_trees, iter_spans_only)
        # print(f'S (span only)   P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        print(f'S (span only)   F1:{f1_s:.2%}')
        if f1_s > max_f1_S:
            max_f1_S = f1_s
            max_f1_S_epoch = epoch + 1
        
        p, r, f1_n = eval_trees(pred_trees, gold_trees, iter_nuclearity_spans)
        # print(f'N (span + dir)  P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        print(f'N (span + dir)  F1:{f1_n:.2%}')
        if f1_n > max_f1_N:
            max_f1_N = f1_n
            max_f1_N_epoch = epoch + 1     

        p, r, f1_r = eval_trees(pred_trees, gold_trees, iter_labeled_spans)
        # print(f'R (span + label)        P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        print(f'R (span + label)        F1:{f1_r:.2%}')
        if f1_r > max_f1_R:
            max_f1_R = f1_r
            max_f1_R_epoch = epoch + 1

        p, r, f1 = eval_trees(pred_trees, gold_trees, iter_labeled_spans_with_nuclearity)
        # print(f'F (full)        P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        print(f'F (full)        F1:{f1:.2%}')
        if f1 > max_f1_F:
            max_f1_F = f1
            max_f1_F_epoch = epoch + 1
        if f1 > saved_model_f1_f:
            #we decide whether or not save the model based on Full F1, but save best scores from each component
            if config.DEBUG == False:
                model.save(model_path)
            saved_model_f1_s = f1_s
            saved_model_f1_n = f1_n
            saved_model_f1_r = f1_r
            saved_model_f1_f = f1

    print("\n--------------------------------------------------------")
    print("Saved model:\n--------------------------------------------------------")
    print("F1 span:\t", saved_model_f1_s)
    print("F1 span + direction:\t", saved_model_f1_n)
    print("F1 span + relation:\t", saved_model_f1_r)
    print("F1 full:\t", saved_model_f1_f)
    


    print("\n--------------------------------------------------------")
    print("Best epochs:\n--------------------------------------------------------")
    print("metric\t|\tscore\t|\tepoch")
    print("max f1 (span):\t", max_f1_S, "\t", max_f1_S_epoch)
    print("max f1 (span + dir):\t", max_f1_N, "\t", max_f1_N_epoch)
    print("max f1 (span + rel):\t", max_f1_R, "\t", max_f1_R_epoch)
    print("max f1 (span + rel + dir):\t", max_f1_F, "\t", max_f1_F_epoch)
    print("--------------------------------------------------------")

    assert saved_model_f1_f == max_f1_F
    # return these if we want to get averages from last epoch
    # return f1_span, f1_n, f1_r, f1
    # return saved (best full f1) model scores
    return saved_model_f1_s, saved_model_f1_n, saved_model_f1_r, saved_model_f1_f
    
        

if __name__ == '__main__':

    start_time = time.time()
    random_seeds = config.RANDOM_SEEDS
    if config.DEBUG == True:
        r_seed = random_seeds[0]
        random.seed(r_seed)
        torch.manual_seed(r_seed)
        torch.cuda.manual_seed(r_seed)
        np.random.seed(r_seed)
        main(None)

    else:

        #create dir for the experiment
        experiment_dir_path = os.path.join(config.OUTPUT_DIR, "experiment" + str(config.EXPERIMENT_ID) + "-" + config.EXPERIMENT_DESCRIPTION + "-" + str(date.today()))
        if not os.path.exists(experiment_dir_path):
            os.makedirs(experiment_dir_path)

        #copy the config file into the experiment directory
        shutil.copyfile(config.CONFIG_FILE, os.path.join(experiment_dir_path, "config.py"))

        with open(os.path.join(experiment_dir_path, "log"), "w") as f:
            sys.stdout = f
        
            span_scores = np.zeros(len(random_seeds))
            nuclearity_scores = np.zeros(len(random_seeds)) # span + direction
            relations_scores = np.zeros(len(random_seeds)) # span + relation label
            full_scores = np.zeros(len(random_seeds)) # span + direction + relation label

            best_f1_full = 0    # best Full F1 among the runs with different seeds
            best_seed = random_seeds[0] # the random seed that produced the best Full F1

            # do training for every random seed
            for i in range(len(random_seeds)):
                r_seed = random_seeds[i]
                print("===============")
                print("random seed ", r_seed)
                print("---------------")
                
                random.seed(r_seed)
                torch.manual_seed(r_seed)
                torch.cuda.manual_seed(r_seed)
                np.random.seed(r_seed)

                rs_results = main(experiment_dir_path)

                span_scores[i] = rs_results[0]
                nuclearity_scores[i] = rs_results[1]
                relations_scores[i] = rs_results[2]
                full_scores[i] = rs_results[3]

                # if the full f1 output from the random seed is higher than previously recorded best f1 (from a diff seed), 
                # update the best f1 and the random seed
                if rs_results[3] > best_f1_full:
                    best_f1_full = rs_results[3]
                    best_seed = r_seed


            print("\n========================================================")
            print(f"Mean scores from {len(random_seeds)} runs with different random seeds (the scores are from the saved model, i.e., best model based on full f1 score):")
            print("--------------------------------------------------------")
            print("F1 (span):\t", np.around(np.mean(span_scores), decimals=3), "±", np.around(np.std(span_scores), decimals=3))
            print("F1 (span + dir):\t", np.around(np.mean(nuclearity_scores), decimals=3), "±", np.around(np.std(nuclearity_scores), decimals=3))
            print("F1 (span + rel):\t", np.around(np.mean(relations_scores), decimals=3), "±", np.around(np.std(relations_scores), decimals=3))
            print("F1 (full):\t", np.around(np.mean(full_scores), decimals=3), "±", np.around(np.std(full_scores), decimals=3))
            print("Best random seed:\t", best_seed)
            print("Time it took to run the script --- %s seconds ---" % (time.time() - start_time))


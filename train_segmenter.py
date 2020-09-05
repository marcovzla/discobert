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

def main(experiment_dir_path):
    # if config.DEBUG == False:
    model_dir_path = os.path.join(experiment_dir_path, "rs" + str(r_seed))
    # print("model dir path: ", model_dir_path)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    model_path = os.path.join(model_dir_path, config.SEGMENTER_MODEL_FILENAME)
    print("path to model: ", model_path)

    device = torch.device('cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu')
    model = SegmentationModel()
    model.to(device)

    # load data and split in train and validation sets
    train_ds, valid_ds = train_test_split(list(load_annotations(config.TRAIN_PATH)), test_size=config.TEST_SIZE)

    # for td in train_ds:
    #     print("raw: ", td.raw)
    #     print("edus", td.edus)
    
    if config.SORT_INPUT == True:
        # construct new train_ds
        train_ids_by_length = {}
        for item in train_ds:
            train_ids_by_length.setdefault(len(item.edus), []).append(item)

        train_ds = []
        for n in sorted(train_ids_by_length):
            for ann in train_ids_by_length[n]:
                train_ds.append(ann)

    if config.SORT_VALIDATION == True:
        # construct new train_ds
        valid_ids_by_length = {}
        for item in valid_ds:
            valid_ids_by_length.setdefault(len(item.edus), []).append(item)

        valid_ds = []
        for n in sorted(valid_ids_by_length):
            for ann in valid_ids_by_length[n]:
                valid_ds.append(ann)

    num_training_steps = int(len(train_ds[:50]) * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters(model), lr=config.LR, eps=1e-8, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # the scores from the model that was best based on Full F1 score and that was saved
    saved_model_f1 = 0
    
    # max component scores and the associated epochs
    # Note: saved_model_f1_f should be equal to max_f1_F
    max_f1 = 0
    max_f1_epoch = 1
    
    for epoch in range(config.SEGMENT_EPOCHS):
        if epoch > 0: print()
        print("-----------")
        print(f'epoch: {epoch+1}/{config.SEGMENT_EPOCHS}')
        print("-----------")
        segmenter_engine.train_fn(train_ds[:50], model, optimizer, device, scheduler)
        pred_trees, gold_trees = segmenter_engine.eval_fn(valid_ds, model, device)
        p, r, f1 = eval_boundaries(pred_trees, gold_trees)
        print(f'boundaries   P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        # print(f'S (span only)   F1:{f1_s:.2%}')
        if f1 > max_f1:
            max_f1 = f1
            max_f1_epoch = epoch + 1
        
        if f1 > saved_model_f1:
            # if config.DEBUG == False:
            model.save(model_path)
            saved_model_f1 = f1
    
    # print("\n--------------------------------------------------------")
    # print("Saved model:\n--------------------------------------------------------")
    # print("F1 span:\t", saved_model_f1_s)
    # print("F1 span + direction:\t", saved_model_f1_n)
    # print("F1 span + relation:\t", saved_model_f1_r)
    # print("F1 full:\t", saved_model_f1_f)
    


    # print("\n--------------------------------------------------------")
    # print("Best epochs:\n--------------------------------------------------------")
    # print("metric\t|\tscore\t|\tepoch")
    # print("max f1 (span):\t", max_f1_S, "\t", max_f1_S_epoch)
    # print("max f1 (span + dir):\t", max_f1_N, "\t", max_f1_N_epoch)
    # print("max f1 (span + rel):\t", max_f1_R, "\t", max_f1_R_epoch)
    # print("max f1 (span + rel + dir):\t", max_f1_F, "\t", max_f1_F_epoch)
    # print("--------------------------------------------------------")

    assert saved_model_f1 == max_f1
    # return these if we want to get averages from last epoch
    # return f1_span, f1_n, f1_r, f1
    # return saved (best full f1) model scores
    return saved_model_f1
    
        

if __name__ == '__main__':

    print("Printing out config settings:")
    print("debug: ", config.DEBUG)
    print("encoding: ", config.ENCODING)
    print("tokenizer: ", config.SEGMENTER_TOKENIZER)
    print("sort input: ", config.SORT_INPUT)
    print("test size: ", config.TEST_SIZE)

    start_time = time.time()
    random_seeds = config.RANDOM_SEEDS
    if config.DEBUG == True:
        r_seed = random_seeds[0]
        random.seed(r_seed)
        torch.manual_seed(r_seed)
        torch.cuda.manual_seed(r_seed)
        np.random.seed(r_seed)
        main(os.path.join(config.SEGMENTER_OUTPUT_DIR, "from_debug" + "-" + str(date.today())))

    else:

        #create dir for the experiment
        experiment_dir_path = os.path.join(config.SEGMENTER_OUTPUT_DIR, "experiment" + str(config.EXPERIMENT_ID) + "-" + config.EXPERIMENT_DESCRIPTION + "-" + str(date.today()))
        if not os.path.exists(experiment_dir_path):
            os.makedirs(experiment_dir_path)

        #copy the config file into the experiment directory
        shutil.copyfile(config.CONFIG_FILE, os.path.join(experiment_dir_path, "config.py"))

        with open(os.path.join(experiment_dir_path, "log"), "w") as f:
            sys.stdout = f
            print("Printing out config settings:")
            print("Separate rel and dir classifiers (true=3-classifier parser): ", config.SEPARATE_ACTION_AND_DIRECTION_CLASSIFIERS)
            print("debug: ", config.DEBUG)
            print("encoding: ", config.ENCODING)
            print("tokenizer: ", config.TOKENIZER)
            # print("model: ", config.MODEL)
            print("use attention", config.USE_ATTENTION)
            print("use relation and dir emb-s: ", config.INCLUDE_RELATION_EMBEDDING, " ", config.INCLUDE_DIRECTION_EMBEDDING)
            print("sort input: ", config.SORT_INPUT)
            print("test size: ", config.TEST_SIZE)
            scores = np.zeros(len(random_seeds))
            
            best_f1 = 0    # best Full F1 among the runs with different seeds
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

                scores[i] = rs_results
                

                # if the full f1 output from the random seed is higher than previously recorded best f1 (from a diff seed), 
                # update the best f1 and the random seed
                if rs_results > best_f1:
                    best_f1 = rs_results
                    best_seed = r_seed



            score = np.around(np.mean(scores), decimals=3)
            score_sd = np.around(np.std(scores), decimals=3)
            
            
            print("\n========this print out is to check if i made any mistakes adding the code here from train==============================")
            print(f"Mean scores from {len(random_seeds)} runs with different random seeds:")
            print("--------------------------------------------------------")
            print("F1 (span):\t", score, "±", score_sd) # this is f1; todo: add p and r
            textpm_string = "\\\\textpm".replace("\\\\", "\\")
            # print("latex pringout: ", f" & {span_score} {textpm_string} {span_score_sd} &  {nuc_score} {textpm_string} {nuc_score_sd} &  {rel_score} {textpm_string} {rel_score_sd} & {full_score} {textpm_string} {full_score_sd} \\\\")

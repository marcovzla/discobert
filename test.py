#!/usr/bin/env python

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from model import DiscoBertModel
from rst import load_annotations, iter_spans_only, iter_nuclearity_spans, iter_labeled_spans, iter_labeled_spans_with_nuclearity, iter_label_and_direction, iter_labels
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

def get_label_nuclearity_distribution(predictions):
    label_list = [
            "None", 
            "attribution",
        "background",
        "cause",
        "comparison",
        "condition",
        "contrast",
        "elaboration",
        "enablement",
        "evaluation",
        "explanation",
        "joint",
        "manner_means",
        "same_unit",
        "summary",
        "temporal",
        "textual_organization",
        "topic_change",
    "topic_comment"]

    nuclearity_label_dict = {relation:{"None":0, "LeftToRight":0, "RightToLeft":0} for i,relation in enumerate(label_list)}
    nuclearity_label_none_compatible = {"none_compatible": []}
    for doc_pred in predictions:
        for pred in doc_pred:
            # print(pred)
            pred_split = pred.split("::")
            label = pred_split[0]
            nuclearity = pred_split[1]
            if nuclearity == "None":
                # print(nuclearity_label_dict[label])
                nuclearity_label_dict[label]["None"] += 1
                if not label in nuclearity_label_none_compatible["none_compatible"]:
                    nuclearity_label_none_compatible["none_compatible"].append(label)
            elif nuclearity == "RightToLeft":
                nuclearity_label_dict[label]["RightToLeft"] += 1
            elif nuclearity == "LeftToRight":
                nuclearity_label_dict[label]["LeftToRight"] += 1
            else:
                print("something went horribly wrong: ", item)

    return nuclearity_label_dict



def eval_trees(pred_trees, gold_trees, view_fn):
    if "iter_labels" in str(view_fn):
        # this is to get confusion-matrix-like information
        all_pred_spans = [f'{x}' for t in pred_trees for x in view_fn(t.get_nonterminals())]
        all_gold_spans = [f'{x}' for t in gold_trees for x in view_fn(t.get_nonterminals())]
    else:
        all_pred_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in pred_trees]
        all_gold_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in gold_trees]
    # print("view fn", view_fn)

    # per_doc_scores = [prf1(pred, gold) for pred, gold in zip(all_pred_spans, all_gold_spans)]
    # # print("PER DOC SCORES: ", per_doc_scores)
    # for score in per_doc_scores:
    #     print(score)
    #For the following two, make sure to set "rerun_dev_eval" to True
    # to compare how 
    if "iter_label_and_direction" in str(view_fn):
        print("predicted:")
        pred_dict = get_label_nuclearity_distribution(all_pred_spans)
        print("gold:")
        gold_dict = get_label_nuclearity_distribution(all_gold_spans)
        for key in pred_dict:
            print("label: ", key, "\npred: ", pred_dict[key], "\ngold: ", gold_dict[key], "\n")

    

    confusion = {}
    # print(str(view_fn))
    if "iter_labels" in str(view_fn):
        for i, gold_label in enumerate(all_gold_spans):
            predicted = all_pred_spans[i]
            # print(gold_label, " ", predicted)
            if gold_label in confusion:
                if predicted in confusion[gold_label]:
                    confusion[gold_label][predicted] += 1
                else:
                    confusion[gold_label][predicted] = 1
            else:
                confusion[gold_label] = {gold_label: 1}

    for gold_label in confusion.keys():
        print("==============\n", gold_label, "\n--------------")
        predictions_dict = {k: v for k, v in sorted(confusion[gold_label].items(), key=lambda item: item[1], reverse=True)}
        for predicted in predictions_dict:
            print(predicted, "\t", confusion[gold_label][predicted])


    tpfpfns = [tpfpfn(pred, gold) for pred, gold in zip(all_pred_spans, all_gold_spans)]
    # print(tpfpfns)
    tp, fp, fn = np.array(tpfpfns).sum(axis=0)
    # print(tp, fp, fn)
    scores = calc_prf_from_tpfpfn(tp, fp, fn)
    # scores = np.array(scores).mean(axis=0).tolist()
    # print(scores)
    return scores

# def eval_trees(pred_trees, gold_trees, view_fn):
    # all_pred_spans = [f'{x}' for t in pred_trees for x in view_fn(t.get_nonterminals())]
    # all_gold_spans = [f'{x}' for t in gold_trees for x in view_fn(t.get_nonterminals())]
#     scores = prf1(all_pred_spans, all_gold_spans)
#     return scores

def main(path_to_model, test_ds, threshold):
    print("TH: ", threshold)
    
    if config.RERUN_DEV_EVAL == True:
        # print("START LOAD TRAIN/DEV SET")
        train_ds, valid_ds = train_test_split(list(load_annotations(config.TRAIN_PATH)), test_size=config.TEST_SIZE)
    device = torch.device('cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu')
    model = DiscoBertModel.load(path_to_model)
    model.to(device)
    
    if config.RERUN_DEV_EVAL == True:
        if config.SORT_INPUT:
            valid_ids_by_length = {}
            for item in valid_ds:
                valid_ids_by_length.setdefault(len(item.edus), []).append(item)

            valid_ds = []
            for n in sorted(valid_ids_by_length):
                for ann in valid_ids_by_length[n]:
                    valid_ds.append(ann)

        pred_trees, gold_trees = engine.eval_fn(valid_ds, model, device, threshold=threshold)
    else:
        pred_trees, gold_trees = engine.eval_fn(test_ds, model, device, threshold=threshold)

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

        # eval_trees(pred_trees, gold_trees, iter_label_and_direction) #this is to get label distributions
        # eval_trees(pred_trees, gold_trees, iter_labels) # this is to get confusion-matrix-like outputs
        

        return f1_s, f1_n, f1_r, f1
    else:
        return None
    
if __name__ == '__main__':

    test_ds = list(load_annotations(config.VALID_PATH))

    experiment_dir_path = config.OUTPUT_DIR/config.EXPERIMENT_DESCRIPTION
    random_seeds = config.RANDOM_SEEDS
    if config.PRINT_TREES == False:
        if config.RERUN_DEV_EVAL == True:
            log_name = "eval_log_dev_noise_experiment.txt" 
        else:
            log_name = "eval_log_noise_experiment"
        with open(os.path.join(experiment_dir_path, log_name), "w") as f:
            sys.stdout = f
            
            thresholds = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.50]
            span_scores = np.zeros(len(thresholds))
            
            nuclearity_scores = np.zeros(len(thresholds)) # span + direction
            relations_scores = np.zeros(len(thresholds)) # span + relation label
            full_scores = np.zeros(len(thresholds)) # span + direction + relation label

            for i in range(len(random_seeds)):
                r_seed = random_seeds[i]
                random.seed(r_seed)
                torch.manual_seed(r_seed)
                torch.cuda.manual_seed(r_seed)
                np.random.seed(r_seed)
                
                for j, th in enumerate(thresholds):
                    print("J: ", j, " th: ", th)
                    path_to_model = os.path.join(str(experiment_dir_path/'rs') + str(r_seed), config.MODEL_FILENAME)
                    print("model path: ", path_to_model)
                    rs_results = main(path_to_model, test_ds, th)
                    print(rs_results)
                    span_scores[j] = rs_results[0]
                    nuclearity_scores[j] = rs_results[1]
                    relations_scores[j] = rs_results[2]
                    full_scores[j] = rs_results[3]
                    print(span_scores)
            print(len(span_scores))
            for i, th in enumerate(thresholds):
                print(str(th) + "\t" + str(span_scores[i]))

            print("-----")
            for th in thresholds:
                print(th)

            print("-----")

            for score in span_scores:
                print(score)

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
            
            log_name = "trees_with_doc_id" + str(r_seed) + ".txt"
            with open(os.path.join(experiment_dir_path, log_name), "w") as f:
                sys.stdout = f
                rs_results = main(path_to_model, test_ds)
            
                


        
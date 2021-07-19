#!/usr/bin/env python

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from model import DiscoBertModel
from rst import load_annotations, load_gold_lin2019_annotations_for_testing, load_one_sent_lin2019_data_for_processing, iter_spans_only, iter_nuclearity_spans, iter_labeled_spans, iter_labeled_spans_with_nuclearity, iter_label_and_direction, iter_labels
from utils import prf1, tpfpfn, calc_prf_from_tpfpfn
from model_glove import DiscoBertModelGlove
from model_glove_2_class import DiscoBertModelGlove2Class
import config
import engine, segmenter_engine
import random
import os
import sys
import shutil
from datetime import date
import time
from segmenter_model import SegmentationModel
from utils import make_word2index

def optimizer_parameters(model):
    no_decay = ['bias', 'LayerNorm']
    named_params = list(model.named_parameters())
    return [
        {'params': [p for n,p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n,p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]


def get_label_nuclearity_distribution(predictions):
    label_list = config.ID_TO_LABEL
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


def eval_tree_pools(pred_trees, gold_trees, view_fn):
    
    all_pred_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in pred_trees]
    # print("pred: ", all_pred_spans)
    # for t in pred_trees:
    #     print("t: ", t.text)
    all_gold_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in gold_trees]
    
    current_pred_pool = []
    current_gold_pool = []

    for i, tree_pred in enumerate(all_pred_spans):
        current_pred_pool.append(tree_pred)
        current_gold_pool.append(all_gold_spans[i])
        assert len(current_pred_pool) == len(current_gold_pool)
        max_len = 0
        for doc in current_gold_pool:
            # print("len doc: ", len(doc))
            if len(doc) > max_len:
                max_len = len(doc)
        # print("max len: ", max_len)


        tpfpfns = [tpfpfn(pred, gold) for pred, gold in zip(current_pred_pool, current_gold_pool)]
        # print(tpfpfns)
        tp, fp, fn = np.array(tpfpfns).sum(axis=0)
        # print(tp, fp, fn)
        scores = calc_prf_from_tpfpfn(tp, fp, fn)
        print("len of pool: ", len(current_pred_pool), "; max document length (in spans): ", max_len, "score: ", scores[0])
        # scores = np.array(scores).mean(axis=0).tolist()
        # print(scores)
        # return scores


def eval_trees(pred_trees, gold_trees, view_fn, pred_edus=None, gold_edus=None):
    # print("EVAL")
    # if config.ONE_SENT_EVAL:
        
    # pred_edus - edus from our segmenter
    if config.USE_SEGMENTER:
        all_pred_spans = [[f'{x}' for x in view_fn(t.get_nonterminals(), pred_edus[idx])] for idx, t in enumerate(pred_trees)]
        all_gold_spans = [[f'{x}' for x in view_fn(t.get_nonterminals(), gold_edus[idx])] for idx, t in enumerate(gold_trees)]
    else:
        all_pred_spans = [[f'{x}' for x in view_fn(t.get_nonterminals(), None)] for  t in pred_trees]
        if config.ONE_SENT_EVAL:
            all_gold_spans = load_gold_lin2019_annotations_for_testing(config.ONE_SENT_DATA_DIR)
        else:
            all_gold_spans = [[f'{x}' for x in view_fn(t.get_nonterminals(), None)] for  t in gold_trees]

    # print("HERE 1")

    for i in all_pred_spans:
        print("+>", i)
    relation2instances = {}
    # if view_fn == iter_labeled_spans:
    for per_doc in all_pred_spans:
        for item in per_doc:
            # print(item)
            label = item.split("::")[-1]
            if label in relation2instances:
                relation2instances[label] += 1
            else:
                relation2instances[label] = 1

    for w in sorted(relation2instances, key=relation2instances.get, reverse=True):
        print(w, relation2instances[w])
    # for i in range(len(list(all_gold_spans))):
    #     for j in range(len(list(all_gold_spans))):
    #         print(">>", list(all_gold_spans)[i][j], "||", list(all_pred_spans)[i][j])
    # print(">>", list(all_pred_spans)[:10])
    # print(">>>>", list(all_gold_spans)[:10])
    # for sp in all_gold_spans:
    #     print(">>>", sp)
    # if "iter_labels" in str(view_fn):
    #     # this is to get confusion-matrix-like information
    #     all_pred_spans = [f'{x}' for t in pred_trees for x in view_fn(t.get_nonterminals())]
    #     all_gold_spans = [f'{x}' for t in gold_trees for x in view_fn(t.get_nonterminals())]
    # else:
    #     all_pred_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in pred_trees]
    #     all_gold_spans = [[f'{x}' for x in view_fn(t.get_nonterminals())] for t in gold_trees]

    # per_doc_scores = [prf1(pred, gold) for pred, gold in zip(all_pred_spans, all_gold_spans)]
    # # print("PER DOC SCORES: ", per_doc_scores)
    # for score in per_doc_scores:
    #     print(score)

    #For the following two, make sure to set "rerun_dev_eval" to True
    # to compare how 
    # if "iter_label_and_direction" in str(view_fn):
    #     print("predicted:")
    #     pred_dict = get_label_nuclearity_distribution(all_pred_spans)
    #     print("gold:")
    #     gold_dict = get_label_nuclearity_distribution(all_gold_spans)
    #     for key in pred_dict:
    #         print("label: ", key, "\npred: ", pred_dict[key], "\ngold: ", gold_dict[key], "\n")

    

    # confusion = {}
    # # print(str(view_fn))
    # if "iter_labels" in str(view_fn):
    #     for i, gold_label in enumerate(all_gold_spans):
    #         predicted = all_pred_spans[i]
    #         # print(gold_label, " ", predicted)
    #         if gold_label in confusion:
    #             if predicted in confusion[gold_label]:
    #                 confusion[gold_label][predicted] += 1
    #             else:
    #                 confusion[gold_label][predicted] = 1
    #         else:
    #             confusion[gold_label] = {gold_label: 1}

    # for gold_label in confusion.keys():
    #     print("==============\n", gold_label, "\n--------------")
    #     predictions_dict = {k: v for k, v in sorted(confusion[gold_label].items(), key=lambda item: item[1], reverse=True)}
    #     for predicted in predictions_dict:
    #         print(predicted, "\t", confusion[gold_label][predicted])


    tpfpfns = [tpfpfn(pred, gold) for pred, gold in zip(all_pred_spans, all_gold_spans)]
    # print(tpfpfns)
    tp, fp, fn = np.array(tpfpfns).sum(axis=0)
    # print(tp, fp, fn)
    scores = calc_prf_from_tpfpfn(tp, fp, fn)
    # scores = np.array(scores).mean(axis=0).tolist()
    # print(scores)
    return scores   


def main(path_to_model, test_ds, original_test_ds=None):
    # original_test_ds - used when test_ds come from our segmenter
    #set random seed here because the vocab will be built based on the train set
    # random.seed(random_seed)
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # np.random.seed(random_seed)
    
    if config.ENCODING == "glove":
        
        # load data and split in train and validation sets; fixme: how do you do if you don't have a train set? make it based on test?
        train_ds, valid_ds = train_test_split(list(load_annotations(config.TRAIN_PATH)), test_size=config.TEST_SIZE)

        word2index = make_word2index(train_ds)   
        model = DiscoBertModelGlove(word2index).load(path_to_model, word2index)
    elif config.ENCODING == "glove-2-class":
        
        # load data and split in train and validation sets
        train_ds, valid_ds = train_test_split(list(load_annotations(config.TRAIN_PATH)), test_size=config.TEST_SIZE)

        word2index = make_word2index(train_ds)   
        model = DiscoBertModelGlove2Class(word2index).load(path_to_model, word2index)   
    else:
        model = DiscoBertModel.load(path_to_model)
    
    model.to(device)

    pred_trees, gold_trees = engine.eval_fn(test_ds, model, device)

    
    # if config.RERUN_DEV_EVAL == True:
    #     if config.SORT_INPUT:
    #         valid_ids_by_length = {}
    #         for item in valid_ds:
    #             valid_ids_by_length.setdefault(len(item.edus), []).append(item)

    #         valid_ds = []
    #         for n in sorted(valid_ids_by_length):
    #             for ann in valid_ids_by_length[n]:
    #                 valid_ds.append(ann)

    #     pred_trees, gold_trees = engine.eval_fn(valid_ds, model, device)
    # else:
    #     pred_trees, gold_trees = engine.eval_fn(new_test_ds, model, device)

    

    # gold_edu_strings = []
    # print("gold--->", gold_trees[0].gold_spans())
    # for gs in gold_trees[0].gold_spans():
    #     print("gs: ", gs)
    #     start = gs.start
    #     print("start: ", start)
    #     end = gs.stop
    #     print("end: ", end)
    #     edus_start_end = test_ds[0].edus[start:end]
    #     print("edus: ",  edus_start_end)
    #     str_to_compare = edus_start_end[0].replace(" ","")[:5] + edus_start_end[-1].replace(" ","")[-5:]
    #     print("str to compare: " + str_to_compare)
    #     gold_edu_strings.append(str_to_compare)
    # print("pred--->", pred_trees[0].gold_spans())
    # print("gold edu strings: ", gold_edu_strings)
    # for gs in pred_trees[0].gold_spans():
    #     print("gs: ", gs)
    #     start = gs.start
    #     print("start: ", start)
    #     end = gs.stop
    #     print("end: ", end)
    #     edus_start_end = new_test_ds[0].edus[start:end]
    #     print("edus: " , edus_start_end)
    #     str_to_compare = edus_start_end[0].replace(" ","")[:5] + edus_start_end[-1].replace(" ","")[-5:]
    #     if str_to_compare in gold_edu_strings:
    #         print("TRUE")
    #     else:
    #         print("FALSE: " + str_to_compare)

    if config.PRINT_TREES == False:

        # if config.USE_SEGMENTER:
        #     p, r, f1_s = eval_trees(pred_trees, gold_trees, iter_spans_only, test_ds, original_test_ds)
        # else:
        #     p, r, f1_s = eval_trees(pred_trees, gold_trees, iter_spans_only)
        
        
        # #p, r, f1_s = eval_trees(pred_trees, gold_trees, iter_spans_only)
        # print(f'S (span only)   P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        # print(f'S (span only)   F1:{f1_s:.2%}')

        # if config.USE_SEGMENTER:
        #     p, r, f1_n = eval_trees(pred_trees, gold_trees, iter_nuclearity_spans, test_ds, original_test_ds)
        # else:
        #     p, r, f1_n = eval_trees(pred_trees, gold_trees, iter_nuclearity_spans)
           
        # #p, r, f1_n = eval_trees(pred_trees, gold_trees, iter_nuclearity_spans)
        # # print(f'N (span + dir)  P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        # print(f'N (span + dir)  F1:{f1_n:.2%}')
        

        # if config.USE_SEGMENTER:
        #     p, r, f1_r = eval_trees(pred_trees, gold_trees, iter_labeled_spans, test_ds, original_test_ds)
        # else:
        #     p, r, f1_r = eval_trees(pred_trees, gold_trees, iter_labeled_spans)
        
        # #p, r, f1_r = eval_trees(pred_trees, gold_trees, iter_labeled_spans)
        # # print(f'R (span + label)        P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        # print(f'R (span + label)        F1:{f1_r:.2%}')
        
        if config.USE_SEGMENTER:
            print("Use segm")
            p, r, f1 = eval_trees(pred_trees, gold_trees, iter_labeled_spans_with_nuclearity, test_ds, original_test_ds)
        else:

            if config.SENT_EVAL_TYPE=="full":
                p, r, f1 = eval_trees(pred_trees, gold_trees, iter_labeled_spans_with_nuclearity)
            elif config.SENT_EVAL_TYPE == "label":
                p, r, f1 = eval_trees(pred_trees, gold_trees, iter_labeled_spans)
            elif config.SENT_EVAL_TYPE == "nuc":
                p, r, f1 = eval_trees(pred_trees, gold_trees, iter_nuclearity_spans)
            elif config.SENT_EVAL_TYPE == "span":
                p, r, f1 = eval_trees(pred_trees, gold_trees, iter_spans_only)
        #p, r, f1 = eval_trees(pred_trees, gold_trees, iter_labeled_spans_with_nuclearity)
        # print(f'F (full)        P:{p:.2%}\tR:{r:.2%}\tF1:{f1:.2%}')
        print(f'F (full)        F1:{f1:.2%}')

        # eval_trees(pred_trees, gold_trees, iter_label_and_direction) #this is to get label distributions
        # eval_trees(pred_trees, gold_trees, iter_labels) # this is to get confusion-matrix-like outputs
        # eval_tree_pools(pred_trees, gold_trees, iter_labeled_spans_with_nuclearity) #see how scores change with the size of the document

        # return f1_s, f1_n, f1_r, f1
        return None, None, None, f1
    else:
        return None
    
if __name__ == '__main__':


    device = torch.device('cuda' if config.USE_CUDA and torch.cuda.is_available() else 'cpu')
    
    experiment_dir_path = config.OUTPUT_DIR/config.EXPERIMENT_DESCRIPTION
    random_seeds = config.RANDOM_SEEDS
    if config.PRINT_TREES == False:
        log_name = config.LOG_NAME
        with open(os.path.join(experiment_dir_path, log_name), "w") as f:
            sys.stdout = f
            

            span_scores = np.zeros(len(random_seeds))
            nuclearity_scores = np.zeros(len(random_seeds)) # span + direction
            relations_scores = np.zeros(len(random_seeds)) # span + relation label
            full_scores = np.zeros(len(random_seeds)) # span + direction + relation label

            for i in range(len(random_seeds)):
                r_seed = random_seeds[i]
                print(r_seed)
                random.seed(r_seed)
                torch.manual_seed(r_seed)
                torch.cuda.manual_seed(r_seed)
                np.random.seed(r_seed)
                path_to_model = os.path.join(str(experiment_dir_path/'rs') + str(r_seed), config.MODEL_FILENAME)
                

                if config.ONE_SENT_EVAL:
                    test_ds = list(load_one_sent_lin2019_data_for_processing(config.ONE_SENT_DATA_DIR))
                    print(test_ds[0], "<<")
                    print(len(test_ds[0].edus), "<<<")
                else:
                    if config.RERUN_DEV_EVAL == True:
                        # print("START LOAD TRAIN/DEV SET")
                        train_ds, test_ds = train_test_split(list(load_annotations(config.TRAIN_PATH)), test_size=config.TEST_SIZE)
                        
                    else:
                        test_ds = list(load_annotations(config.VALID_PATH))

                # print(test_ds[0].edus)
                if config.SORT_INPUT == True:
                    # construct new train_ds
                    test_ids_by_length = {}
                    for item in test_ds:
                        test_ids_by_length.setdefault(len(item.edus), []).append(item)

                    test_ds = []
                    for n in sorted(test_ids_by_length):
                        for ann in test_ids_by_length[n]:
                            test_ds.append(ann)

                # for the sake of dev eval; have to resegment every time
                if config.USE_SEGMENTER:
                    segm_experiment_dir_path = config.SEGMENTER_OUTPUT_DIR/config.SEGMENTER_EXPERIMENT_DESCRIPTION
                    segmentaion_model = SegmentationModel.load(os.path.join(str(segm_experiment_dir_path/'rs') + str("22"), config.SEGMENTER_MODEL_FILENAME))
                    segmentaion_model.to(device)
                    segment_test_ds = segmenter_engine.run_fn(test_ds, segmentaion_model, device)

                print("model path: ", path_to_model)
                if config.USE_SEGMENTER:
                    rs_results = main(path_to_model, segment_test_ds, test_ds)
                else:
                    rs_results = main(path_to_model, test_ds)
                # span_scores[i] = rs_results[0]
                # nuclearity_scores[i] = rs_results[1]
                # relations_scores[i] = rs_results[2]
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
            # print("F1 (span):\t", span_score, "±", span_score_sd)
            # print("F1 (span + dir):\t", nuc_score , "±", nuc_score_sd)
            # print("F1 (span + rel):\t", rel_score, "±", rel_score_sd)
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
            
                


        
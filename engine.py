import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from utils import CumulativeMovingAverage
import config
from rst import iter_labeled_spans_with_nuclearity

def train_fn(annotations, model, optimizer, device, scheduler=None, class_weights=None):
    model.train()
    batch_size = 2
    loss_avg = CumulativeMovingAverage()
    # annotations = tqdm(annotations, total=len(annotations))
    # annotations.set_description('train')
    annotation_batches = [annotations[i:i + batch_size] for i in range(0, len(annotations), batch_size)]  
    for batch in annotation_batches:
        batch = tqdm(batch, total=len(batch))
        batch.set_description('train')
        for a in batch:
            loss, tree = model(a.edus, a.dis, class_weights=class_weights)
            loss_avg.add(loss.item())
        batch.set_postfix_str(f'loss={loss_avg:.4f}')
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.zero_grad()

def eval_fn(annotations, model, device, class_weights=None):
    # print("TREES:\n")
    model.eval()
    pred_trees = []
    gold_trees = []
    with torch.no_grad():
        annotations = tqdm(annotations, total=len(annotations))
        annotations.set_description('devel')
        for a in annotations:
            # print(a.docid)
            tree = model(a.edus, annotation=a, class_weights=class_weights)[0]
            pred_trees.append(tree)
            gold_trees.append(a.dis)

            # pred_tree_nodes = iter_labeled_spans_with_nuclearity(tree.get_nonterminals())   #iter_labeled_spans_with_nuclearity
            # print(f"Predicted tree for::{a.docid}")
            # for node in pred_tree_nodes:
            #     print(node)
            # print("\n")

            # gold_tree_nodes = iter_labeled_spans_with_nuclearity(a.dis.get_nonterminals())
            # print(f"Gold tree for::{a.docid}")
            # for node in gold_tree_nodes:
            #     print(node)
            # print("\n")
            
            # print(f"Predicted tree for::{a.docid}::\n", tree.to_nltk(), "\n")
            # print(f"Gold tree for::{a.docid}::\n", a.dis.to_nltk(), "\n")
    return pred_trees, gold_trees

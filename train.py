import argparse
import os
import torch
from torch.optim import Adam
from tqdm.autonotebook import tqdm
from model import DiscoBertModel
from rst import load_annotations, iter_spans_only
from transformers import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--val')
    parser.add_argument('--modeldir')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    return args

# load some parameters
epochs = 20
learning_rate = 1e-3
device = torch.device('cpu')

SAVED_MODELS_PATH = '/work/bsharp/data/discobert/RST/models/debug-distil'
RST_CORPUS_PATH = '/work/bsharp/data/discobert/RST/data/RSTtrees-WSJ-main-1.0/training_subset'
RST_VAL_CORPUS_PATH = '/work/bsharp/data/discobert/RST/data/RSTtrees-WSJ-main-1.0/validation'



# TODO: replace?? tokenizer based on what Enrique said OR pass all EDUs at once
# TODO: finish the validation code -- ability to eval just tree, tree + direction, tree + dir + label
# TODO: add the classifiers for label and direction


def train(num_epochs, learning_rate, device, train_dir, val_dir, model_dir):
    with open(os.path.join(model_dir, "log.txt"), 'a') as logfile:

        torch.cuda.empty_cache()
        discobert = DiscoBertModel()
        discobert.set_device(device, init_weights=True)
        discobert.to(device)

        # setup the optimizer, loss, etc
        optimizer = Adam(params=discobert.parameters(), lr=learning_rate)

        # for each epoch
        for epoch_i in range(num_epochs):
            print(f'Beginning epoch {epoch_i}')
            print(f'Beginning epoch {epoch_i}', file=logfile)

            for annotation in tqdm(list(load_annotations(train_dir))):
                discobert.zero_grad()
                loss, pred_tree = discobert(annotation.edus, annotation.dis)
                loss.backward()
                optimizer.step()

            print(f'Finished epoch {epoch_i}')
            print(f'Finished epoch {epoch_i}', file=logfile)

            # save model
            epoch_model_dir = os.path.join(model_dir, f'discobert_{epoch_i}')
            if not os.path.exists(epoch_model_dir):
                os.makedirs(epoch_model_dir)
            discobert.save_pretrained(epoch_model_dir)
            # evaluate on validation
            if discobert.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            predict(val_dir, discobert, logfile)


def predict(data_dir, discobert, logfile):
    all_gold_nodes = []
    all_pred_nodes = []
    all_gold_spans = []
    all_pred_spans = []

    for annotation in tqdm(list(load_annotations(data_dir))):
        pred_tree = discobert(annotation.edus)[0]

        ann_gold_nodes = annotation.dis.get_nonterminals()
        ann_gold_spans = [f'{annotation.docid}_{x}' for x in list(iter_spans_only(ann_gold_nodes))]
        all_gold_nodes.extend(ann_gold_nodes)
        all_gold_spans.extend(ann_gold_spans)

        ann_pred_nodes = pred_tree.get_nonterminals()
        ann_pred_spans = [f'{annotation.docid}_{x}' for x in list(iter_spans_only(ann_pred_nodes))]
        all_pred_nodes.extend(ann_pred_nodes)
        all_pred_spans.extend(ann_pred_spans)

    p, r, f1 = eval(all_gold_spans, all_pred_spans)
    print(f'P:{p}\tR:{r}\tF1:{f1}')
    print(f'P:{p}\tR:{r}\tF1:{f1}', file=logfile)


def eval(gold, pred):
    TP, FP, FN = 0, 0, 0
    for g in gold:
        if g in pred:
            TP += 1
        else:
            FN += 1

    for p in pred:
        if p not in gold:
            FP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, f1


if __name__=='__main__':

    args = parse_args()
    train(args.epochs, args.lr, args.device, args.train, args.val, args.modeldir)


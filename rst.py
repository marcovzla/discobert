import os
from collections import namedtuple
from nltk import Tree

RST_CORPUS_PATH = 'RST/data/RSTtrees-WSJ-main-1.0/TEST/'

Annotation = namedtuple('Annotation', 'raw dis edus')

def load_annotation(name, corpus=RST_CORPUS_PATH):
    raw_path = os.path.join(corpus, name)
    dis_path = raw_path + '.dis'
    edus_path = raw_path + '.edus'
    raw = load_raw(raw_path)
    dis = load_dis(dis_path)
    edus = load_edus(edus_path)
    return Annotation(raw, dis, edus)

def load_raw(name):
    with open(name) as f:
        return f.read()

def load_dis(name):
    with open(name) as f:
        return Tree.fromstring(f.read())

def load_edus(name):
    with open(name) as f:
        return f.readlines()

import os
import re
from collections import namedtuple

RST_CORPUS_PATH = 'RST/data/RSTtrees-WSJ-main-1.0/TRAINING/'

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
        return Node.from_string(f.read())

def load_edus(name):
    with open(name) as f:
        return [line.strip() for line in f]


class Node:

    def __init__(self, kind):
        self.kind = kind
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def lhs(self):
        if len(self.children) == 2:
            return self.children[0]

    def rhs(self):
        if len(self.children) == 2:
            return self.children[1]

    @classmethod
    def from_string(cls, string):
        key, tree, pos = parse_node(tokenize(string), 0)
        propagate_labels(tree)
        return tree


def tokenize(data):
    token_specification = [
        ('OPEN_PARENS',  r'\('),
        ('CLOSE_PARENS', r'\)'),
        ('STRING',       r'_!.*?_!'),
        ('SYMBOL',       r'[\w-]+'),
    ]
    token_regex = '|'.join(f'(?P<{name}>{regex})' for name,regex in token_specification)
    return list(re.finditer(token_regex, data))

def parse_node(tokens, position):
    i = position
    t = tokens[i]
    kind = t.lastgroup
    value = t.group()
    if value == 'leaf':
        n = int(tokens[i+1].group()) - 1 # zero-based
        assert tokens[i+2].lastgroup == 'CLOSE_PARENS'
        return ('leaf', n, i+3)
    elif value == 'span':
        n = int(tokens[i+1].group()) - 1 # zero-based
        m = int(tokens[i+2].group()) # exclusive on the end
        assert tokens[i+3].lastgroup == 'CLOSE_PARENS'
        return ('span', [n,m], i+4)
    elif value == 'rel2par':
        label = tokens[i+1].group()
        assert tokens[i+2].lastgroup == 'CLOSE_PARENS'
        return ('rel2par', label, i+3)
    elif value == 'text':
        text = tokens[i+1].group()[2:-2]
        assert tokens[i+2].lastgroup == 'CLOSE_PARENS'
        return ('text', text, i+3)
    elif kind == 'OPEN_PARENS':
        # skip
        return parse_node(tokens, i + 1)
    else:
        node = Node(value)
        pos = i + 1
        while tokens[pos].lastgroup != 'CLOSE_PARENS':
            key, val, pos = parse_node(tokens, pos)
            if key in ['Nucleus', 'Satellite']:
                node.children.append(val)
            else:
                setattr(node, key, val)
        return (value, node, pos+1)

def propagate_labels(node):
    if node.is_leaf():
        return
    [lhs, rhs] = node.children
    if lhs.kind == 'Nucleus' and rhs.kind == 'Satellite':
        direction = 'LeftToRight'
        label = rhs.rel2par
    elif lhs.kind == 'Satellite' and rhs.kind == 'Nucleus':
        direction = 'RightToLeft'
        label = lhs.rel2par
    elif lhs.kind == rhs.kind:
        direction = 'None'
        label = lhs.rel2par
    else:
        raise Exception('this should be impossible')
    node.label = label
    node.direction = direction
    propagate_labels(lhs)
    propagate_labels(rhs)

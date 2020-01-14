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
        return TreeNode.from_string(f.read())

def load_edus(name):
    with open(name) as f:
        return [line.strip() for line in f]

class TreeNode:

    def __init__(self, kind=None, children=None, text=None, leaf=None, span=None, rel2par=None):
        self.kind = kind
        self.children = children if children is not None else []
        self.text = text
        self.leaf = leaf
        self.span = span

    def is_leaf(self):
        return len(self.children) == 0

    @property
    def lhs(self):
        if len(self.children) == 2:
            return self.children[0]

    @property
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
    if kind == 'OPEN_PARENS':
        # skip
        return parse_node(tokens, i + 1)
    elif value == 'leaf':
        # the index of the corresponding EDU
        n = int(tokens[i+1].group()) - 1 # zero-based
        assert tokens[i+2].lastgroup == 'CLOSE_PARENS'
        return (value, n, i+3)
    elif value == 'span':
        # the span of EDUs that are leaves to this node
        n = int(tokens[i+1].group()) - 1 # zero-based
        m = int(tokens[i+2].group()) # exclusive end
        assert tokens[i+3].lastgroup == 'CLOSE_PARENS'
        return (value, range(n, m), i+4)
    elif value == 'rel2par':
        # the discourse relation label.
        # this should be in the parent node,
        # which is the reason for having propagate_labels()
        label = tokens[i+1].group()
        assert tokens[i+2].lastgroup == 'CLOSE_PARENS'
        return (value, label, i+3)
    elif value == 'text':
        # the EDU's text
        text = tokens[i+1].group()[2:-2] # drop _! delimiters
        assert tokens[i+2].lastgroup == 'CLOSE_PARENS'
        return (value, text, i+3)
    elif value in ['Nucleus', 'Satellite', 'Root']:
        # a tree node
        node = TreeNode(kind=value)
        pos = i + 1
        while tokens[pos].lastgroup != 'CLOSE_PARENS':
            key, val, pos = parse_node(tokens, pos)
            if key in ['Nucleus', 'Satellite']:
                node.children.append(val)
            else:
                setattr(node, key, val)
        return (value, node, pos+1)
    else:
        raise Exception(f"unrecognized kind '{kind}'")

def propagate_labels(node):
    """propagate rel2par labels from children to parent"""
    # are we done?
    if node.is_leaf():
        return
    # unpack children
    [lhs, rhs] = node.children
    # find label and direction
    if lhs.kind == 'Nucleus' and rhs.kind == 'Satellite':
        label = rhs.rel2par
        direction = 'LeftToRight'
    elif lhs.kind == 'Satellite' and rhs.kind == 'Nucleus':
        label = lhs.rel2par
        direction = 'RightToLeft'
    elif lhs.kind == rhs.kind:
        label = lhs.rel2par
        direction = 'None'
    else:
        raise Exception('unexpected children kinds')
    # set label and direction
    node.label = label
    node.direction = direction
    # recurse
    propagate_labels(lhs)
    propagate_labels(rhs)

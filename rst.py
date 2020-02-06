from glob import glob
import os
import re
from copy import copy, deepcopy
from collections import namedtuple

Annotation = namedtuple('Annotation', 'raw dis edus')
LEFT_TO_RIGHT = 'LeftToRight'
RIGHT_TO_LEFT = 'RightToLeft'

def load_annotations(directory):
    pattern = os.path.join(directory, '*.dis')
    for filename in glob(pattern):
        raw_path = os.path.splitext(filename)[0]
        # print("file:", raw_path)
        yield load_annotation(raw_path)

def load_annotation(raw_path):
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

# primarily for evaluation -- make a "view" of the tree nodes to compare
def iter_spans_only(treenodes):
    for t in treenodes:
        yield t.span

class TreeNode:

    def __init__(self, kind=None, children=None, text=None, leaf=None, span=None, rel2par=None, label=None, direction=None, embedding=None):
        self.kind = kind # ['Nucleus', 'Satellite', 'Root']
        self.children = children if children is not None else []
        self.text = text
        if leaf is not None and span is not None:
            raise AttributeError("???")
        self.leaf = leaf # Int -- the index of the EDU *if* it is a leaf node, else None
        self.span = range(leaf, leaf+1) if leaf is not None else span # Range of EDU indexes
        self.label = label # relation label
        self.direction = direction # leftToRight, rightToLeft, or None
        self.embedding = embedding

    def __eq__(self, other):
        return self.span == other.span and self.label == other.label and self.direction == other.direction and self.children == other.children

    def __deepcopy__(self, memodict={}):
        newone = TreeNode()
        newone.kind = deepcopy(self.kind, memodict)
        newone.children = deepcopy(self.children, memodict)
        newone.text = deepcopy(self.text, memodict)
        newone.leaf = deepcopy(self.leaf, memodict)
        newone.span = deepcopy(self.span, memodict)
        newone.label = deepcopy(self.label, memodict)
        newone.direction = deepcopy(self.direction, memodict)
        # we don't copy embedding on purpose
        return newone

    @property
    def is_terminal(self):
        return len(self.children) == 0

    @property
    def lhs(self):
        if len(self.children) == 2:
            return self.children[0]

    @property
    def rhs(self):
        if len(self.children) == 2:
            return self.children[1]

    def get_nonterminals(self):
        return list(self.iter_nonterminals())

    def get_terminals(self):
        return list(self.iter_terminals())

    def calc_span(self):
        edus = [t.leaf for t in self.iter_terminals()]
        self.span = range(min(edus), max(edus) + 1)

    def iter_nodes(self):
        # preorder traversal
        yield self
        for c in self.children:
            for n in c.iter_nodes():
                yield n

    def iter_nonterminals(self):
        for n in self.iter_nodes():
            if not n.is_terminal:
                yield n

    def iter_terminals(self):
        for n in self.iter_nodes():
            if n.is_terminal:
                yield n

    def gold_spans(self):
        golds = [self.span]
        if not self.is_terminal:
            for c in self.children:
                golds.extend(c.gold_spans())
        return golds


    @classmethod
    def from_string(cls, string):
        key, tree, pos = parse_node(tokenize(string), 0)
        propagate_labels(tree)
        binarize_tree(tree)
        return tree


def tokenize(data):
    token_specification = [
        ('OPEN_PARENS',  r'\('),
        ('CLOSE_PARENS', r'\)'),
        ('STRING',       r'_!.*?_!'),
        ('SYMBOL',       r'[\w-]+'),
        ('COMMENT',      r'//.*')
    ]
    token_regex = '|'.join(f'(?P<{name}>{regex})' for name,regex in token_specification)
    return [tok for tok in re.finditer(token_regex, data) if tok.lastgroup != 'COMMENT']


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
        # set correct span
        node.calc_span()
        return (value, node, pos+1)
    else:
        raise Exception(f"unrecognized kind '{kind}' value='{value}'")

def propagate_labels(node):
    """propagate rel2par labels from children to parent"""
    # are we done?
    if node.is_terminal:
        return
    # unpack children
    [lhs, *other_children, rhs] = node.children
    # find label and direction
    if lhs.kind == 'Nucleus' and rhs.kind == 'Satellite':
        label = rhs.rel2par
        direction = LEFT_TO_RIGHT
    elif lhs.kind == 'Satellite' and rhs.kind == 'Nucleus':
        label = lhs.rel2par
        direction = RIGHT_TO_LEFT
    elif lhs.kind == rhs.kind:
        label = lhs.rel2par
        direction = 'None'
    else:
        raise Exception('unexpected children kinds')
    # set label and direction
    node.label = label
    node.direction = direction
    # recurse
    for c in node.children:
        propagate_labels(c)

def binarize_tree(node):
    if node.is_terminal:
        return node
    if len(node.children) > 2:
        old_children = node.children
        node.children = []
        node.children.append(old_children[0])
        node.children.append(make_binary_child(node, old_children[1:]))
    for c in node.children:
        binarize_tree(c)
    return node

def make_binary_child(parent, children):
    if len(children) == 1:
        return children[0]
    node = copy(parent)
    node.children = [children[0], make_binary_child(parent, children[1:])]
    node.calc_span()
    return node

from glob import glob
import os
import re
from copy import copy, deepcopy
from collections import namedtuple
from nltk import Tree
from utils import extractrelation

Annotation = namedtuple('Annotation', 'docid raw dis edus')
LEFT_TO_RIGHT = 'LeftToRight'
RIGHT_TO_LEFT = 'RightToLeft'

def load_annotations(directory):
    pattern = os.path.join(directory, '*.dis')
    for filename in glob(pattern):
        raw_path = os.path.splitext(filename)[0]
        # print("file:", raw_path)
        yield load_annotation(raw_path)

def load_annotation(raw_path):
    base_name = os.path.basename(raw_path)
    dis_path = raw_path + '.dis'
    edus_path = raw_path + '.edus'
    raw = load_raw(raw_path)
    dis = load_dis(dis_path)
    edus = load_edus(edus_path)
    return Annotation(base_name, raw, dis, edus)

def load_raw(name):
    with open(name) as f:
        return f.read()

def load_dis(name):
    with open(name) as f:
        return TreeNode.from_string(f.read())

def load_edus(name):
    with open(name) as f:
        return [line.strip() for line in f]
        

def get_span_text(t, annot): #one tree node and a set of annotations(?) for the document
    start = t.span.start 
    end = t.span.stop 
    # text = "".join(annot.edus[start:end]).replace(" ", "").lower() #
    # edus_start_end = annot.edus[start:end]
    text = edus_start_end[0].replace(" ","")[:5] + edus_start_end[-1].replace(" ","")[-5:]
    return text

# primarily for evaluation -- make a "view" of the tree nodes to compare (S)
def iter_spans_only(treenodes, annot=None):
    if annot != None:
        for t in treenodes:
            # print("t span: ", t.span)
            # print("TEXT: ",  text)
            text = get_span_text(t, annot)
            yield text
    else:
        for t in treenodes:
            yield t.span


# for evaluation, must get the span and the nuclearity right (N)
def iter_nuclearity_spans(treenodes, annot=None):
    if annot == None:
        for t in treenodes:
            yield f'{t.span}::{t.direction}'
    else:
        for t in treenodes:
            text = get_span_text(t, annot)
            yield f'{text}::{t.direction}'

# for evaluation -- must get span and relation label right (R)
def iter_labeled_spans(treenodes, annot=None):
    if annot == None:
        for t in treenodes:
            yield f'{t.span}::{t.label}'
    else:
        for t in treenodes:
            text = get_span_text(t, annot)
            yield f'{text}::{t.label}'

# for evaluation -- must get everything right (F)
def iter_labeled_spans_with_nuclearity(treenodes, annot=None):
    if annot == None:
        for t in treenodes:
            yield f'{t.span}::{t.direction}::{t.label}'
    else:
        for t in treenodes:
            text = get_span_text(t, annot)
            yield f'{text}::{t.direction}::{t.label}'


# this is just for checking the predicted label distribution 
def iter_label_and_direction(treenodes, annot=None):
    if annot == None:
        for t in treenodes:
            yield f'{t.label}::{t.direction}'

def iter_labels(treenodes, annot=None):
    if annot == None:
        for t in treenodes:
            yield f'{t.label}'

def make_offsets(text, tokens):
    """given some raw text and its corresponding tokens,
    this function returns the token's character offsets"""
    start = 0
    for tok in tokens:
        start = text.index(tok, start)
        stop = start + len(tok)
        yield (start, stop)
        start = stop

class TreeNode:

    #todo: method dump trees
    #stopping condiction and recursive; if no children, then it's a leaf, then print leaf 
    #when it's not a leaf (non_term), print the info for this node and then call this method again for each of its children

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
        # self.edu_offsets = edu_offsets

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

    # def to_nltk(self):
    #     if self.is_terminal:
    #         return Tree('EDU', [self.text])
    #     else:
    #         return Tree(self.label, [self.lhs.to_nltk(), self.rhs.to_nltk()])

    def to_nltk(self):
        if self.is_terminal:
            # return Tree('TEXT', [" ".join(self.text.replace("(", "[").replace(")", "]").split(" ")[:5])])
            # return Tree('TEXT', [])
            return Tree('TEXT', [self.text.replace("(", "[").replace(")", "]")])
        else:
            if self.direction == 'LeftToRight':
                label = f'{self.label}:NS'
            elif self.direction == 'RightToLeft':
                label = f'{self.label}:SN'
            else:
                label = self.label
        return Tree(label, [self.lhs.to_nltk(), self.rhs.to_nltk()])

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
    # print("token in parse node: ", t)
    kind = t.lastgroup
    value = t.group()
    # print("t group (value): ", value)
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
        # print("text: ", text)
        assert tokens[i+2].lastgroup == 'CLOSE_PARENS'
        return (value, text, i+3)
    # elif value 
    elif value in ['Nucleus', 'Satellite', 'Root']:
        # a tree node
        node = TreeNode(kind=value)
        text = None
        # print("NODE: ", node)
        pos = i + 1
        
        while tokens[pos].lastgroup != 'CLOSE_PARENS':
            key, val, pos = parse_node(tokens, pos)
            # print("key, val, pos: ", key, " ", val, " ", pos)
            if key in ['Nucleus', 'Satellite']:
                node.children.append(val)
            elif key == "text":
                # print("TRUE")
                # print("val in key = text: ", val)
                text = val
                
            else:
                setattr(node, key, val)
        # set correct span
        node.calc_span()
        # print("text: ", text)
        # node.text = text
        # print("node text: ", node.text)
        # print("node children: ", node.children)
        return (value, node, pos+1)
    else:
        raise Exception(f"unrecognized kind '{kind}' value='{value}'")

def propagate_labels(node):
    """propagate rel2par labels from children to parent"""
    # are we done?
    # print("Node text in prop labels: ", node.text)
    # print("node label in prop labels: ", node.label)
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
    node.label = extractrelation(label)
    node.direction = direction
    # recurse
    for c in node.children:
        propagate_labels(c)

def binarize_tree(node):
    # print(len(node.children))
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

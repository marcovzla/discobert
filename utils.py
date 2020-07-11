import config
import numpy as np

class CumulativeMovingAverage:
    # https://en.wikipedia.org/wiki/Moving_average#Cumulative_moving_average

    def __init__(self):
        self.n = 0
        self.avg = 0.0

    def __format__(self, format_spec):
        return format(self.avg, format_spec)

    def add(self, x):
        self.avg = float(x) + self.n * self.avg
        self.n += 1
        self.avg /= self.n

def prf1(pred, gold):
    tp, fp, fn = 0, 0, 0
    for g in gold:
        if g in pred:
            tp += 1
        else:
            fn += 1
    for p in pred:
        if p not in gold:
            fp += 1
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * ((p * r) / (p + r))
    return p, r, f1

# from https://github.com/jiyfeng/DPLP/blob/master/code/util.py
# code from Ji and Eisenstein paper
def extractrelation(s):
    """ Extract discourse relation on different level
    """
    coarse_map = {
        "none": ["none"],
        "attribution": ["attribution"],
        "background": ["background", "circumstance"],
        "cause": ["cause", "consequence", "result"],
        "comparison": ["analogy", "comparison", "preference", "proportion"],
        "condition": ["condition", "contingency", "hypothetical", "otherwise"],
        "contrast": ["antithesis", "concession", "contrast"],
        "elaboration": ["definition", "elaboration", "example"],
        "enablement": ["enablement", "purpose"],
        "evaluation": ["comment", "comment_e", "conclusion", "evaluation", "interpretation"],
        "explanation": ["evidence", "explanation", "reason"],
        "joint": ["disjunction", "joint", "list"],
        "manner_means": ["manner", "means"],
        "same_unit": ["same_unit"],
        "summary": ["summary", "restatement"],
        "temporal": ["inverted", "sequence", "temporal"],
        "textual_organization": ["textualorganization"],
        "topic_change": ["topic_drift", "topic_shift"],
        "topic_comment": ["comment_topic", "topic_comment", "problem", "question", "rhetorical", "statement"],
    }
    fg_to_coarse = {
        "none": "none",
        "attribution": "attribution",
        "background": "background",
        "cause": "cause",
        "consequence": "cause",
        "result": "cause",
        "analogy": "comparison",
        "comparison": "comparison",
        "preference": "comparison",
        "proportion": "comparison",
        "condition": "condition",
        "contingency": "condition",
        "hypothetical": "condition",
        "otherwise": "condition",
        "antithesis": "contrast",
        "concession": "contrast",
        "contrast": "contrast",
        "definition": "elaboration",
        "elaboration": "elaboration",
        "example": "elaboration",
        "enablement": "enablement",
        "purpose": "enablement",
        "circumstance": "background",
        "comment": "evaluation",
        "comment_e": "evaluation",
        "conclusion": "evaluation",
        "evaluation": "evaluation",
        "interpretation": "evaluation",
        "evidence": "explanation",
        "explanation": "explanation",
        "reason": "explanation",
        "disjunction": "joint",
        "joint": "joint",
        "list": "joint",
        "manner": "manner_means",
        "means": "manner_means",
        "same_unit": "same_unit",
        "summary": "summary",
        "restatement": "summary",
        "inverted": "temporal",
        "sequence": "temporal",
        "temporal": "temporal",
        "textualorganization": "textual_organization",
        "topic_drift": "topic_change",
        "topic_shift": "topic_change",
        "comment_topic": "topic_comment",
        "topic_comment": "topic_comment",
        "problem": "topic_comment",
        "question": "topic_comment",
        "rhetorical": "topic_comment",
        "statement": "topic_comment",
    }
    items = s.lower().split('-')
    if items[0] == 'same':
        rela = '_'.join(items[:2])
    # We added
    elif items[0] == "topic":
        rela = '_'.join(items[:2]) 
    elif items[0] == "comment":
        rela = '_'.join(items[:2]) 
    else:
        rela = items[0]
    return fg_to_coarse[rela]

# def load_glove(path):
#     glove = {}
#     emb_size = 0
#     with open(path, 'rb') as f:
#         for l in f:
#             if list(f).index(l) == 0:
#                 emb_size = int(l.decode().split("")[-1]
#             else:
#             # split lines
#                 line = l.decode().split()
#                 # first part is word
#                 word = line[0]
#                 # the rest is the embeddings
#                 vec = np.array(line[1:]).astype(float)
#                 # feed dict
#                 glove[word] = vec
#     return glove, emb_size

# def make_vocabulary_and_emb(edus):
#     """make word vocabulary from a list of sentences"""
#     glove, emb_size = load_glove(config.GLOVE_PATH)
#     print("emb size: ", emb_size)
#     tokenizer = config.TOKENIZER
#     vocab = {}
#     for edu in edus:
#         words = tokenizer(edu.raw)
#         for word in words:
#             if not word in vocab:
#                 if word in glove:
#                     vocab[word] = glove(word)
#     vocab["<UNK>"] = np.random.normal(scale=0.6, size=emb_size)
#     print("vocab: ", vocab)
#     print("vocab length: ", len(vocab))
#     return vocab



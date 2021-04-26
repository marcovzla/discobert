from pathlib import Path
import tokenizers
from tokenizers import Tokenizer 
from transformers import *
import torchtext
from torchtext.data import get_tokenizer


ENCODING = 'roberta' #options in this branch (~~'ed ones are not yet ready): ~~"glove"~~, "glove-2-class", ~~'glove-2-class-stack-only'~~, and (some of) the encodings at the end of the file
USE_SEGMENTER = False # has to be true for training segmenter
SEGMENTER_ENCODING = 'bert' 
NO_CONNECTIVES = False # to mask discourse markers (full list below), set to True; only implemented for 2 class version
DEBUG = False # no saving of files; output in the terminal; first random seed from the list
RERUN_DEV_EVAL = False # True to rerun eval on the same dev sets that were used during training
EXPERIMENT_ID = 29
EXPERIMENT_DESCRIPTION = "roberta-just-class-weights-dropcls-sort-input" # enter a brief description that will make the experiment easy to identify
SEGMENTER_EXPERIMENT_DESCRIPTION = "experiment3-segmenter-with-all-dev-examples-2021-04-02" # used to write and read a segmenter model
SEGMENTER_RS_TO_USE=137 #whichever rs was best during training (based on dev)
LOG_NAME = "log" # have been using "log" for training and "eval_log" for testing, and "eval_log_dev" for rerunning eval on dev set
PRINT_TREES = False
TEST_SIZE = 0.15 #If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25. (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
EPOCHS = 30
SEGMENT_EPOCHS = 20
MAX_LEN = 50
DROPOUT = 0.2
USE_CUDA = True
LR = 3e-5 #default for bert: 3e-5; default for glove: .01

RANDOM_SEEDS = [22, 42, 137, 198, 202]
HIDDEN_SIZE = 200
RELATION_LABEL_HIDDEN_SIZE = 5 #10
DIRECTION_HIDDEN_SIZE = 10
USE_CLASS_WEIGHTS = True # class weights for relation label classifier

INCLUDE_RELATION_EMBEDDING = False
INCLUDE_DIRECTION_EMBEDDING = False #has to be false for the two classifier version
USE_ATTENTION = False # not currently in model_glove
DROP_CLS = True #whether or not drop the beginning of sequence token (bos_token)
SORT_INPUT = True  #simplified curriculum learning
SORT_VALIDATION = False


CONNECTIVES =  ["accordingly","additionally","after","afterward","also","alternatively","although","and","as","as a result","as an alternative","as if","as long as","as soon as","as though","as well","because","before","before and after","besides","but","by comparison","by contrast","by then","consequently","conversely","earlier","either", "or","else","except","finally","for","for example","for instance","further","furthermore","hence","however","if","if and when","in addition","in contrast","in fact","in other words","in particular","in short","in sum","in the end","in turn","indeed","insofar as","instead","later","lest","likewise","meantime","meanwhile","moreover","much as","neither", "nevertheless","next","nonetheless","nor","now that","on the contrary","on the one hand", "on the other hand","on the other hand","once","or","otherwise","overall","plus","previously","rather","regardless","separately","similarly","simultaneously","since","so","so that","specifically","still","then","thereafter","thereby","therefore","though","thus","till","ultimately","unless","until","when","when and if","whereas","while","yet"]

#hpc paths
DISCOBERT_PATH = Path('/groups/bsharp/alexeeva/discourse/data/discobert/').expanduser() 
DISCOBERT_CODE_PATH = Path('/groups/bsharp/alexeeva/discourse/discobert').expanduser()

#saturno paths
#DISCOBERT_PATH = Path('~/data/discobert').expanduser() 
#DISCOBERT_CODE_PATH = Path('~/discobert').expanduser()
OUTPUT_DIR = DISCOBERT_CODE_PATH/'outputs'
SEGMENTER_OUTPUT_DIR = DISCOBERT_CODE_PATH/'segmenter_outputs'
TRAIN_PATH = DISCOBERT_PATH/'RSTtrees-WSJ-main-1.0'/'TRAINING'
VALID_PATH = DISCOBERT_PATH/'RSTtrees-WSJ-main-1.0'/'TEST'

MODEL_FILENAME = 'discobert.model'

SEGMENTER_MODEL_FILENAME = 'segmenter.model'
CONFIG_FILE = DISCOBERT_CODE_PATH/'config.py' # this file will be copies to each experiment directory for record keeping
SEPARATE_ACTION_AND_DIRECTION_CLASSIFIERS = False #ATTN: if False, INCLUDE_DIRECTION_EMBEDDING has to be False
if SEPARATE_ACTION_AND_DIRECTION_CLASSIFIERS == True:
    ID_TO_ACTION = ['shift', 'reduce']
else:
    ID_TO_ACTION = ['shift', 'reduceL', 'reduceR', 'reduce']
 

# this was used in the glove branch; keeping for debugging
# if ENCODING == "glove-2-class":
#     ID_TO_ACTION = ['shift', 'reduceL', 'reduceR', 'reduce']
# else:
#     ID_TO_ACTION = ['shift', 'reduce']


ACTION_TO_ID = {action:i for i,action in enumerate(ID_TO_ACTION)}

ID_TO_SEGMENT_CLASSES = ["B", "I"]
SEGMENT_CLASS_TO_ID = {action:i for i,action in enumerate(ID_TO_SEGMENT_CLASSES)}

ID_TO_DIRECTION = ['None', 'LeftToRight', 'RightToLeft']
DIRECTION_TO_ID = {direction:i for i,direction in enumerate(ID_TO_DIRECTION)}

ID_TO_LABEL = [
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
    "topic_comment",
]


LABEL_TO_ID = {relation:i for i,relation in enumerate(ID_TO_LABEL)}

if USE_SEGMENTER==True:
    SEGMENTER_BERT_PATH = DISCOBERT_PATH/('bert-base-cased')
    SEGMENTER_TOKENIZER = tokenizers.BertWordPieceTokenizer(str(SEGMENTER_BERT_PATH/'vocab.txt'), lowercase=False)
    SEGMENTER_TOKENIZER.enable_padding() #max_length=MAX_LEN)


if ENCODING == "bert":
    # "pre-trained using a combination of masked language modeling objective and next sentence prediction" (https://huggingface.co/transformers/model_doc/bert.html)
    # outputs last hidden state and pooled output
    # has CLS (bos_token) token
    BERT_PATH = DISCOBERT_PATH/('bert-base-cased')
    # SEGMENTER_TOKENIZER = BertTokenizer(str(BERT_PATH/'vocab.txt'), lowercase=False)
    TOKENIZER = tokenizers.BertWordPieceTokenizer(str(BERT_PATH/'vocab.txt'), lowercase=False)
    TOKENIZER.enable_padding() #max_length=MAX_LEN)
if ENCODING == "bert-large":
    BERT_PATH = DISCOBERT_PATH/('bert-large-cased')
    TOKENIZER = tokenizers.BertWordPieceTokenizer(str(BERT_PATH/'vocab.txt'), lowercase=False)
    TOKENIZER.enable_padding() #max_length=MAX_LEN)
elif ENCODING == "roberta": 
    # "builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates" (https://huggingface.co/transformers/model_doc/roberta.html)
    # "raw hidden-states without any specific head on top"
    BERT_PATH = DISCOBERT_PATH/('roberta-base')
    TOKENIZER = RobertaTokenizer.from_pretrained(str(BERT_PATH))
elif ENCODING == "openai-gpt":
    # "pre-trained using language modeling on a large corpus will long range dependencies. [...] trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. " (https://huggingface.co/transformers/model_doc/gpt.html)
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    BERT_PATH = DISCOBERT_PATH/('openai-gpt')
    TOKENIZER = OpenAIGPTTokenizer.from_pretrained(str(BERT_PATH))
    TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})
elif ENCODING == "gpt2":
    # "pre-trained using language modeling on a large corpus will long range dependencies. [...] trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. " (https://huggingface.co/transformers/model_doc/gpt.html)
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    BERT_PATH = DISCOBERT_PATH/('gpt2')
    TOKENIZER = GPT2Tokenizer.from_pretrained(str(BERT_PATH))
    TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})
elif ENCODING == "xlnet":
    # "pre-trained using an autoregressive method to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence factorization order" (https://huggingface.co/transformers/model_doc/xlnet.html)
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    # has CLS (bos_token) token
    BERT_PATH = DISCOBERT_PATH/('xlnet-base-cased')
    TOKENIZER = XLNetTokenizer.from_pretrained(str(BERT_PATH), mem_len=1024)
elif ENCODING == "distilbert":
    # "trained by distilling Bert base." "Knowledge distillation [...] is a compression technique in which a compact model - the student - is trained to reproduce the behaviour of a larger model - the teacher -or an ensemble of models" (https://arxiv.org/abs/1910.01108).
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    # pretty sure this has a cls token
    BERT_PATH = DISCOBERT_PATH/('distilbert-base-uncased')
    TOKENIZER = DistilBertTokenizer.from_pretrained(str(BERT_PATH))
elif ENCODING == "albert":
    # "similar to bert, but with a few tweaks [...] Next sentence prediction is replaced by a sentence ordering prediction" (https://huggingface.co/transformers/model_summary.html#albert)
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    # has CLS token
    BERT_PATH = DISCOBERT_PATH/('albert-base-v2')
    TOKENIZER = AlbertTokenizer.from_pretrained(str(BERT_PATH))
elif ENCODING == "ctrl": # does not work: returns None for some (unk?) tokens
    # "CTRL was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. Leveraging this feature allows CTRL to generate syntactically coherent text" (https://huggingface.co/transformers/model_doc/ctrl.html#ctrltokenizer)
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    BERT_PATH = DISCOBERT_PATH/('ctrl')
    TOKENIZER = CTRLTokenizer.from_pretrained(str(BERT_PATH))
    TOKENIZER.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>'})
elif ENCODING == "glove" or ENCODING == "glove-2-class" or ENCODING == "glove-2-class-stack-only":
    BERT_PATH = None
    EMBEDDING_SIZE = 50
    GLOVE_PATH = "/groups/bsharp/alexeeva/discourse/data/glove/vectors.txt"
    TOKENIZER = get_tokenizer("basic_english")


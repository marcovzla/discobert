from pathlib import Path
import tokenizers
from tokenizers import Tokenizer 
from transformers import RobertaTokenizer
import torchtext
from torchtext.data import get_tokenizer

ENCODING = 'glove-2-class' #other options in this branch: "glove-2-class", "bert", and "roberta"
NO_CONNECTIVES = True # to mask discourse markers (full list below), set to True; only implemented for 2 class version
DEBUG = True # no saving of files; output in the terminal; first random seed from the list
EXPERIMENT_ID = 27
EXPERIMENT_DESCRIPTION = "experiment27-GloveEmbedding-two-classifier-train-dev-based-on-rs-15-percent-dev-default-settings-no-connectives-2020-08-09" # enter a brief description that will make the experiment easy to identify
TEST_SIZE = 0.15 #If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25. (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
EPOCHS = 30
MAX_LEN = 50
DROPOUT = 0.2
USE_CUDA = True
LR = .01 #default for bert: 3e-5; default for glove: .01

RANDOM_SEEDS = [22, 42, 137, 198, 202]
HIDDEN_SIZE = 200
RELATION_LABEL_HIDDEN_SIZE = 50
DIRECTION_HIDDEN_SIZE = 20

INCLUDE_RELATION_EMBEDDING = False
INCLUDE_DIRECTION_EMBEDDING = False
USE_ATTENTION = False # not currently in model_glove
DROP_CLS = False
SORT_INPUT = False #simplified curriculum learning

CONNECTIVES =  ["accordingly","additionally","after","afterward","also","alternatively","although","and","as","as a result","as an alternative","as if","as long as","as soon as","as though","as well","because","before","before and after","besides","but","by comparison","by contrast","by then","consequently","conversely","earlier","either", "or","else","except","finally","for","for example","for instance","further","furthermore","hence","however","if","if and when","in addition","in contrast","in fact","in other words","in particular","in short","in sum","in the end","in turn","indeed","insofar as","instead","later","lest","likewise","meantime","meanwhile","moreover","much as","neither", "nevertheless","next","nonetheless","nor","now that","on the contrary","on the one hand", "on the other hand","on the other hand","once","or","otherwise","overall","plus","previously","rather","regardless","separately","similarly","simultaneously","since","so","so that","specifically","still","then","thereafter","thereby","therefore","though","thus","till","ultimately","unless","until","when","when and if","whereas","while","yet"]


DISCOBERT_PATH = Path('~/data/discobert').expanduser() 
DISCOBERT_CODE_PATH = Path('~/discobert').expanduser()
OUTPUT_DIR = DISCOBERT_CODE_PATH/'outputs'
TRAIN_PATH = DISCOBERT_PATH/'RSTtrees-WSJ-main-1.0'/'TRAINING'
VALID_PATH = DISCOBERT_PATH/'RSTtrees-WSJ-main-1.0'/'TEST'

MODEL_FILENAME = 'discobert.model'

CONFIG_FILE = DISCOBERT_CODE_PATH/'config.py' # this file will be copies to each experiment directory for record keeping

if ENCODING == "glove-2-class":
    ID_TO_ACTION = ['shift', 'reduceL', 'reduceR', 'reduce']
else:
    ID_TO_ACTION = ['shift', 'reduce']

ACTION_TO_ID = {action:i for i,action in enumerate(ID_TO_ACTION)}

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


if ENCODING == "bert":
    BERT_PATH = DISCOBERT_PATH/('bert-base-cased')
    TOKENIZER = tokenizers.BertWordPieceTokenizer(str(BERT_PATH/'vocab.txt'), lowercase=False)
    TOKENIZER.enable_padding() #max_length=MAX_LEN)
elif ENCODING == "roberta":
    BERT_PATH = DISCOBERT_PATH/('roberta-base')
    TOKENIZER = RobertaTokenizer.from_pretrained(str(BERT_PATH))
elif ENCODING == "glove" or ENCODING == "glove-2-class":
    BERT_PATH = None
    EMBEDDING_SIZE = 50
    GLOVE_PATH = "/home/alexeeva/data/glove/vectors.txt"
    TOKENIZER = get_tokenizer("basic_english")


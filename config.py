from pathlib import Path
import tokenizers
from tokenizers import Tokenizer 
from transformers import *
# from transformers import AutoTokenizer, AutoModelWithLMHead

DEBUG = False # no saving of files; output in the terminal; first random seed from the list
EXPERIMENT_ID = 15
EXPERIMENT_DESCRIPTION = "xlnet-trying-to-reproduce-the-error-5-epochs" # enter a brief description that will make the experiment easy to identify, e.g., "Original-run" means with the default settings before any tweaks, e.g., attention or relation embedding, were added 
LATEST_COMMIT = "eb1338c27a197befccf532a9093ab7143476cabd"
TEST_SIZE = 0.25 #If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25. (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
EPOCHS = 5
MAX_LEN = 50
DROPOUT = 0.2
USE_CUDA = True
LR = 3e-5 #default 3e-5

RANDOM_SEEDS = [22, 42, 137, 198, 202]
HIDDEN_SIZE = 200
RELATION_LABEL_HIDDEN_SIZE = 50
DIRECTION_HIDDEN_SIZE = 20

INCLUDE_RELATION_EMBEDDING = False
INCLUDE_DIRECTION_EMBEDDING = False
USE_ATTENTION = False
DROP_CLS = False
SORT_INPUT = False #simplified curriculum learning #this is now just commented out in train.py

DISCOBERT_PATH = Path('~/data/discobert').expanduser() 
DISCOBERT_CODE_PATH = Path('~/discobert').expanduser()
OUTPUT_DIR = DISCOBERT_CODE_PATH/'outputs'
TRAIN_PATH = DISCOBERT_PATH/'RSTtrees-WSJ-main-1.0'/'TRAINING'
VALID_PATH = DISCOBERT_PATH/'RSTtrees-WSJ-main-1.0'/'TEST'
MODEL_FILENAME = 'discobert.model'
CONFIG_FILE = DISCOBERT_CODE_PATH/'config.py' # this file will be copies to each experiment directory for record keeping

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

ENCODING = 'xlnet' 

if ENCODING == "bert":
    # "pre-trained using a combination of masked language modeling objective and next sentence prediction" (https://huggingface.co/transformers/model_doc/bert.html)
    # outputs last hidden state and pooled output
    BERT_PATH = DISCOBERT_PATH/('bert-base-cased')
    TOKENIZER = tokenizers.BertWordPieceTokenizer(str(BERT_PATH/'vocab.txt'), lowercase=False)
    TOKENIZER.enable_padding() #max_length=MAX_LEN)
    # MODEL = BertModel.from_pretrained(str(BERT_PATH))
elif ENCODING == "roberta": 
    # "builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates" (https://huggingface.co/transformers/model_doc/roberta.html)
    # "raw hidden-states without any specific head on top"
    BERT_PATH = DISCOBERT_PATH/('roberta-base')
    TOKENIZER = RobertaTokenizer.from_pretrained(str(BERT_PATH))
    # MODEL = RobertaModel.from_pretrained(str(BERT_PATH))
elif ENCODING == "openai-gpt":
    # "pre-trained using language modeling on a large corpus will long range dependencies. [...] trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. " (https://huggingface.co/transformers/model_doc/gpt.html)
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    BERT_PATH = DISCOBERT_PATH/('openai-gpt')
    TOKENIZER = OpenAIGPTTokenizer.from_pretrained(str(BERT_PATH))
    TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})
    # MODEL = OpenAIGPTModel.from_pretrained(str(BERT_PATH))
elif ENCODING == "xlnet":
    # "pre-trained using an autoregressive method to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence factorization order" (https://huggingface.co/transformers/model_doc/xlnet.html)
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    BERT_PATH = DISCOBERT_PATH/('xlnet-base-cased')
    TOKENIZER = XLNetTokenizer.from_pretrained(str(BERT_PATH))
    MODEL = XLNetModel.from_pretrained(str(BERT_PATH))
elif ENCODING == "distilbert":
    # "trained by distilling Bert base." "Knowledge distillation [...] is a compression technique in which a compact model - the student - is trained to reproduce the behaviour of a larger model - the teacher -or an ensemble of models" (https://arxiv.org/abs/1910.01108).
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    BERT_PATH = DISCOBERT_PATH/('distilbert-base-uncased')
    TOKENIZER = DistilBertTokenizer.from_pretrained(str(BERT_PATH))
    # MODEL = DistilBertModel.from_pretrained(str(BERT_PATH))
elif ENCODING == "albert":
    # "similar to bert, but with a few tweaks [...] Next sentence prediction is replaced by a sentence ordering prediction" (https://huggingface.co/transformers/model_summary.html#albert)
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    BERT_PATH = DISCOBERT_PATH/('albert-base-v2')
    TOKENIZER = AlbertTokenizer.from_pretrained(str(BERT_PATH))
    # MODEL = AlbertModel.from_pretrained(str(BERT_PATH))
elif ENCODING == "ctrl": # does not work: returns None for some (unk?) tokens
    # "CTRL was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. Leveraging this feature allows CTRL to generate syntactically coherent text" (https://huggingface.co/transformers/model_doc/ctrl.html#ctrltokenizer)
    # returns last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
    BERT_PATH = DISCOBERT_PATH/('ctrl')
    TOKENIZER = CTRLTokenizer.from_pretrained(str(BERT_PATH))
    TOKENIZER.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>'})
    # MODEL = CTRLModel.from_pretrained(str(BERT_PATH))



from pathlib import Path
import tokenizers

EPOCHS = 30
MAX_LEN = 50
DROPOUT = 0.2
USE_CUDA = True
LR = 3e-5 #default 3e-5
SEED = 22

DISCOBERT_PATH = Path('~/data/discobert').expanduser()
TRAIN_PATH = DISCOBERT_PATH/'RSTtrees-WSJ-main-1.0'/'TRAINING'
VALID_PATH = DISCOBERT_PATH/'RSTtrees-WSJ-main-1.0'/'TEST'
MODEL_PATH = DISCOBERT_PATH/'discobert.model'

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

BERT_PATH = DISCOBERT_PATH/('bert-base-cased')
TOKENIZER = tokenizers.BertWordPieceTokenizer(str(BERT_PATH/'vocab.txt'), lowercase=False)
TOKENIZER.enable_padding() #max_length=MAX_LEN)
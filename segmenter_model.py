import torch
from torch import nn
from transformers import *
from rst import TreeNode
from transition_system import TransitionSystem
from treelstm import TreeLstm
import config
import torch.nn.functional as F
from collections import namedtuple
from nltk.tokenize import sent_tokenize
import nltk

Annotation = namedtuple('Annotation', 'docid raw dis edus')

inf = float('inf')

def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

class SegmentationModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # config
        self.bert_path = config.BERT_PATH
        self.segmenter_tag_to_id = config.SEGMENT_CLASS_TO_ID
        self.id_to_segmenter_tag = config.ID_TO_SEGMENT_CLASSES       
        self.hidden_size = config.HIDDEN_SIZE
        # init model
        self.encoding = config.ENCODING
        if self.encoding == 'bert':
            self.tokenizer = config.SEGMENTER_TOKENIZER
            self.encoder = BertModel.from_pretrained(self.bert_path)
        elif self.encoding == 'roberta':
            self.tokenizer = config.TOKENIZER
            self.encoder = RobertaModel.from_pretrained(self.bert_path)
        elif self.encoding == 'xlnet':
            self.tokenizer = config.TOKENIZER
            self.encoder = XLNetModel.from_pretrained(self.bert_path)
        self.segment_classifier = nn.Linear(768, 2)
        self.missing_node = nn.Parameter(torch.rand(self.hidden_size, dtype=torch.float))

    @property
    def device(self):
        return self.missing_node.device

    @classmethod
    def load(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, edus, mode, annotation):


        # creating the gold data from the edu input if training or evaluating (not when need to just produce annotations with new edus)
        if mode == "train" or mode == "eval":
            # fixme: these ids may not be needed or only needed for debug printouts; att mask and token type ids should be safe to delete
            ids = []
            # attention_mask = []
            # token_type_ids = []
            gold_tags = [] # gold labels ("B(egin)" or "I(side)") only for the non-special tokens (for bert, excludes [SEP] and [CLS])
            gold_as_tokens = [] # corresponding tokens (for debugging)

            for i in range(len(edus)):

                tokenized_edu = self.tokenizer(edus[i])
                ids.extend(tokenized_edu.input_ids[1:])
                 
                for j in range(len(tokenized_edu.input_ids)): 
                    token_id = tokenized_edu.input_ids[j]
                    token = self.tokenizer.convert_ids_to_tokens([token_id])
                    
                    if token[0] != "[SEP]" and token[0] != "[CLS]":
                        # print("tok added to gold: ", token[0])
                        gold_as_tokens.append(token[0])
                        if j == 1: # with bert encodings starting with the classification (CLS) token, the 1st (not 0th) token should be counted as "B"
                            gold_tags.append("B")   
                        else:
                            gold_tags.append("I")
                            # print("gold O")
            # print("len tok to classify: ", len(tokens_to_classify))
            # print("len tok to classify bertified: ", len(tokens_to_classify_bertified))
            # print("len gold tags: ", len(gold_tags))  
            # print("len gold as tokens: ", len(gold_as_tokens))
            
            # for i, j in enumerate(tokens_to_classify):
            #     print(gold_as_tokens[i], " ", gold_tags[i], tokens_to_classify[i])
        # tokenize raw document text into sentences
        sentences = sent_tokenize(annotation.raw)
        # check if i am not evaluating the right thing---maybe i am feeding the gold data somewhere during eval

        # encoding the sentences
        # fixme: 

        predictions = []        
        losses = []
        potential_edu = [] # only needed for the run mode (only return new annotations, no eval)
        new_edus = []

        current_gold = gold_tags

        predictions_based_on_sent = []

        for i in range(len(sentences)):
            tokenized_sent_not_tensored = self.tokenizer(sentences[i]) # getting tokens from here #fixme: this and next lines need to be unified somehow
            tokenized_sent = self.tokenizer(sentences[i], return_tensors='pt') # getting encodings using this
            sent_tokens = self.tokenizer.convert_ids_to_tokens(tokenized_sent_not_tensored.input_ids)
            # print("bert tokenized sent tokens: ",  tokenizer.convert_ids_to_tokens(tokenized_sent_not_tensored.input_ids))
            # print("bert tokenized sent:", tokenized_sent_not_tensored)

            # prep for the encoder
            ids = torch.tensor(tokenized_sent.input_ids, dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(tokenized_sent.attention_mask, dtype=torch.long).to(self.device)
            token_type_ids = torch.tensor(tokenized_sent.token_type_ids, dtype=torch.long).to(self.device)
            # encode
            encoded_sent, pooled_output = self.encoder(ids, attention_mask, token_type_ids)
            # print('ecnoded sent: ', encoded_sent.shape)
            sent_encoded_only_classifiable_tokens = []
            
            sent_only_classifiable_tokens = []
            # getting classifiable tokens (losing special tokens); squeeze each sent bc for 1 sentence, the 0th dim is 1
            # fixme: get rid of squeezing redundancy
            for i in range(encoded_sent.squeeze(dim=0).shape[0]): 
                if sent_tokens[i] != '[SEP]' and sent_tokens[i] != '[CLS]':
                    if i == 1:
                        predictions_based_on_sent.append("B")
                    else:
                        predictions_based_on_sent.append("I")
                    sent_only_classifiable_tokens.append(sent_tokens[i])
                    # tokens_to_classify.append(sent_tokens[i])
                    token_encoding = encoded_sent.squeeze(dim=0)[i].unsqueeze(dim=0)
                    # print("token encoding: ", token_encoding.shape)
                    # tokens_to_classify_bertified.append(token_encoding)
                    sent_encoded_only_classifiable_tokens.append(token_encoding)
            
            # the first len(sent) predictions are for the current sentence
            gold_for_sent = current_gold[:len(sent_encoded_only_classifiable_tokens)]
            #update 'current_gold', i.e., we have gotten the gold predictions for this sentence, so we can remove them from
            # the gold labels list---the next batch of those is for the next sentence
            current_gold = current_gold[len(sent_encoded_only_classifiable_tokens):]
            # encoded_sentences.append(sent_encoded_only_classifiable_tokens)
            # sentence_tokens_full_doc.append(sent_only_classifiable_tokens)
            num_of_tokens_in_sent = len(sent_encoded_only_classifiable_tokens)

            # go through all the encoded tokens in the sentence
            for j in range(len(sent_encoded_only_classifiable_tokens)):

                if j == 0:
                    # print("j == 0: ", j)
                    predicted_tag = "B-Sent-Init"
                    predictions.append(predicted_tag)
                # get classifier scores
                else:
                    prediction_scores = self.segment_classifier(sent_encoded_only_classifiable_tokens[j])
                    predicted_tag = self.id_to_segmenter_tag[torch.argmax(prediction_scores)]
                    # print("j: ", j)
                    predictions.append(predicted_tag)
                    # are we training?
                if mode == "train":
                    
                    if j == 0:
                        #no need to calc loss
                        continue
                    
                    else:
                        gold_pred_label = gold_for_sent[j]
                        pred_id = self.segmenter_tag_to_id[gold_pred_label]
                        gold_pred = torch.tensor([pred_id], dtype=torch.long).to(self.device)
                        # if self.id_to_segmenter_tag[gold_pred] != predicted_tag:
                        #     print("WRONG PRED: ", self.id_to_segmenter_tag[gold_pred], " ", predicted_tag, " ", tokenizer.convert_ids_to_tokens([ids.squeeze(dim=0)[i]]))
                        # else:
                        #     print("CORRECT PRED: ", self.id_to_segmenter_tag[gold_pred], " ", predicted_tag, " ", tokenizer.convert_ids_to_tokens([ids.squeeze(dim=0)[i]]))
                        
                        # prob dont need this in train?
                        # predictions.append(predicted_tag)
                        
                        loss_on_tags = loss_fn(prediction_scores, gold_pred)

                        losses.append(loss_on_tags)
                elif mode == "eval": 
                    # predictions.append(predicted_tag)

                    # constructing new edus
                    # if found an edu boundary, merge the list of tokens that make up an edu up to this boundary; re-init the potential edu list and add the current token there
                    if predicted_tag == "B" or j == num_of_tokens_in_sent - 1:
                        # print("pot edu: ", potential_edu)
                        potential_edu_str = " ".join(potential_edu).replace(" ##", "") # fixme: what's gonna happen to UNKs? we can't really get those back like this, can we? need to add some other dict to track them?
                        # print("pot edu str: ", potential_edu_str)
                        if len(potential_edu_str) > 0:
                            new_edus.append(potential_edu_str)
                        # re-init potential edu
                        potential_edu = [] 
                        token = sent_only_classifiable_tokens[j]
                        
                        # print("token: ", token)
                        potential_edu.append(token)
                    elif predicted_tag == "I":
                        token = sent_only_classifiable_tokens[j]
                        
                        # print("token: ", token)
                        potential_edu.append(token)
                

                else: #if mode == run (only return new annot, no eval)
                    # constructing new edus
                    # if found an edu boundary, merge the list of tokens that make up an edu up to this boundary; re-init the potential edu list and add the current token there
                    if predicted_tag == "B" or j == num_of_tokens_in_sent - 1:
                        # print("pot edu: ", potential_edu)
                        potential_edu_str = " ".join(potential_edu).replace(" ##", "") # fixme: what's gonna happen to UNKs? we can't really get those back like this, can we? need to add some other dict to track them?
                        # print("pot edu str: ", potential_edu_str)
                        if len(potential_edu_str) > 0:
                            new_edus.append(potential_edu_str)
                        # re-init potential edu
                        potential_edu = [] 
                        token = sent_only_classifiable_tokens[j]
                        
                        # print("token: ", token)
                        potential_edu.append(token)
                    elif predicted_tag == "I":
                        token = sent_only_classifiable_tokens[j]
                        
                        # print("token: ", token)
                        potential_edu.append(token)
                
            
            

        # are we training?
        if mode == "train":
            loss = sum(losses) / (len(losses) - len(sentences)) # subtract the num of sentences bc we did not add loss for the sent-initial edu boundaries
            return loss, new_edus
        elif mode == "eval":
            # print("predictions: ", predictions)
            return predictions, gold_tags
            #to see how we do with only sentence boundaries:
            #return predictions_based_on_sent, gold_tags
        elif mode == "run":
            # this will be for when we need to run the system, but not evaluate --- just return the annotations
            # will need to make a new annotation or replace edus with new_edus in an existing one
            # check if the annotation has been updated! 
            # print("old annotation: ", annotation)
            new_annotation = Annotation(annotation.docid, annotation.raw, annotation.dis, new_edus)
            # print("new annotation: ", annotation)
            
            return Annotation(annotation.docid, annotation.raw, annotation.dis, new_edus), None
        else:
            # need an option that will just take raw and produce annotations with no golds
            NotImplementedError
        # print("outputs: ", outputs)
        # return outputs
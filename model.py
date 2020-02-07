import torch
from torch import nn
from transformers import *
from rst import TreeNode
from transition_system import TransitionSystem

inf = float('inf')

# TODO: document the types of function args

class DiscoBertModel(BertPreTrainedModel):

    def __init__(self, config=BertConfig()):

        super().__init__(config)

        self.num_labels = config.num_labels

        # todo: put bert selection in the config?
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # TODO: attention over whole stack and whole buffer, then this will be 2 * hidden_size
        self.classifier = nn.Linear(3 * config.hidden_size, self.config.num_labels)
        # maybe? -- to merge the nodes during a reduction operation
        self.merger = nn.Linear(2 * config.hidden_size, config.hidden_size)

        # vocabularies
        self.id_to_action = ['shift', 'reduce']
        self.action_to_id = {action: id for id, action in enumerate(self.id_to_action)}

        self.id_to_direction = ['leftToRight', 'rightToLeft', 'None']
        self.direction_to_id = {direction: id for id, direction in enumerate(self.id_to_direction)}

        self.id_to_relation = ['elaboration', 'contrast']  # fixme: add rest/real
        self.relation_to_id = {relation: id for id, relation in enumerate(self.id_to_relation)}



    def set_device(self, device, init_weights=False):
        self.device = device
        # TODO tensor to represent missing node
        self.missing_node = nn.init.normal_(torch.empty(self.config.hidden_size)).to(self.device)
        if init_weights:
            self.init_weights()



    def make_features(self, parser):
        s1 = self.missing_node if len(parser.stack) < 2 else parser.stack[-2].embedding
        s0 = self.missing_node if len(parser.stack) < 1 else parser.stack[-1].embedding
        b = self.missing_node if len(parser.buffer) < 1 else parser.buffer[0].embedding
        # print("s1=", s1.shape)
        # print("s0=", s0.shape)
        # print("b=", b.shape)
        features = torch.cat([s1, s0, b])
        return features


    def merge_embeddings(self, embed_1, embed_2):
        # for now, add
        # return embed_1 + embed_2
        concatted = torch.cat([embed_1, embed_2])
        return nn.ReLU(self.merger(concatted))

    def best_action(self, actions, logits):
        if len(actions) == 1:
            return self.action_to_id[actions[0]]
        elif len(actions) == logits.shape[0]: # FIXME is batch first or after?
            return torch.argmax(logits)
        action_ids = [self.action2id[a] for a in actions]
        mask = torch.ones_like(logits) * -inf
        mask[action_ids] = 0
        masked_logits = logits + mask
        return torch.argmax(masked_logits)

    # forward is for whole document
    def forward(self, edus, gold_tree=None):
        # tokenize
        buffer = []
        for i,edu in enumerate(edus):
            token_ids = self.tokenizer.encode(edu, add_special_tokens=True, return_tensors='pt').to(self.device)
            # self.bert returns a tuple, element 0 is an embedding for each of the words, the second element
            # is an embedding for the whole sentence
            # We squeeze to remove the batch dimension
            embedding = self.bert(token_ids)[1].squeeze()
            buffer.append(TreeNode(text=edu, leaf=i, embedding=embedding))

        # initialize the automata
        parser = TransitionSystem(buffer)

        if gold_tree is not None:
            gold_spans = gold_tree.gold_spans()
            loss_fct = nn.CrossEntropyLoss()
            losses = []

        # if gold_tree is not None:
        #     print("GOLD PATH:")
        #     for step in parser.gold_path(gold_tree):
        #         print(step)
        #     print("-" * 70)

        while not parser.is_done():
            state_features = self.make_features(parser)
            logits = self.classifier(state_features)
            legal_actions = parser.all_legal_actions()
            pred_action = self.best_action(legal_actions, logits)
            if gold_tree is not None:
                gold_steps = parser.all_correct_steps(gold_tree)
                # print('ALL GOLD STEPS:', gold_steps)
                gold_actions = [step.action for step in gold_steps]
                # TODO: should we replace this with getting the gold path from the beginning?
                gold_action = self.best_action(gold_actions, logits)
                # print('GOLD_ACTION:', self.id_to_action[gold_action])
                gold_action = torch.tensor([gold_action]).to(self.device)
                # print("gold_action", gold_action)
                loss = loss_fct(logits.view(-1, self.num_labels), gold_action)
                losses.append(loss)
                # teacher forcing ?
                parser.take_action(self.id_to_action[gold_action], reduce_fn=self.merge_embeddings) # merge_embeddings is only used for REDUCE action
            else:
                parser.take_action(self.id_to_action[pred_action], reduce_fn=self.merge_embeddings) # merge_embeddings is only used for REDUCE action

        # returns the TreeNode for the whole tree
        predicted_tree = parser.get_result()
        outputs = (predicted_tree,)

        if gold_tree is not None:
            loss = sum(losses) / len(losses)
            outputs = (loss,) + outputs
            # todo: precision, recall, f1 of predicted_tree

        return outputs

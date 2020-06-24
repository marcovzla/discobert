from copy import deepcopy
from collections import namedtuple
from rst import TreeNode
import config
import torch
import numpy

Step = namedtuple('Step', 'action label direction')

class TransitionSystem:

    def __init__(self, edus=None):
        self.reset()
        if edus is not None:
            self.buffer.extend(edus)

    def reset(self):
        self.buffer = []
        self.stack = []

    def is_done(self):
        return len(self.buffer) == 0 and len(self.stack) == 1

    def get_result(self):
        if self.is_done():
            return deepcopy(self.stack[0])

    def take_action(self, action, *args, **kwargs):
        if action == 'shift':
            self.shift()
        elif action == 'reduce':
            self.reduce(*args, **kwargs)

    def shift(self):
        node = self.buffer.pop(0)
        self.stack.append(node)

    def reduce(self, label=None, direction=None, reduce_fn=None, rel_tensor=None):
        rhs = self.stack.pop()
        lhs = self.stack.pop()

        # print("label: ", lhs.label)
        # relation_one_hot = torch.FloatTensor(1, len(config.ID_TO_LABEL))
        # print(relation_one_hot)
        # relation_one_hot(1, config.LABEL_TO_ID[label]) = 1.0
        # print(relation_one_hot)
        # print("lhs shape: ", )
        

        
        emb = None if reduce_fn is None else reduce_fn(lhs.embedding, rhs.embedding, rel_tensor) 
        # print(emb)

        node = TreeNode(children=[lhs, rhs], label=label, direction=direction, embedding=emb)
        node.calc_span()
        self.stack.append(node)

    @staticmethod
    def all_actions():
        return ['shift', 'reduce']

    def all_legal_actions(self):
        actions = []
        if self.can_shift():
            actions.append('shift')
        if self.can_reduce():
            actions.append('reduce')
        return actions

    def can_shift(self):
        return len(self.buffer) >= 1

    def can_reduce(self):
        return len(self.stack) >= 2

    def gold_path(self, gold_tree):
        """The correct sequence of steps according to the given gold tree."""
        parser = deepcopy(self)
        while not parser.is_done():
            step = parser.gold_step(gold_tree)
            parser.take_action(*step)
            yield step

    def gold_step(self, gold_tree):
        """The right step to take. If there are several then it returns an arbitrary one."""
        correct_steps = self.all_correct_steps(gold_tree)
        if len(correct_steps) > 0:
            return correct_steps[0]

    def all_correct_steps(self, gold_tree):
        """All steps that would take us from the current state to a state from which it is still possible to reach the gold tree."""
        correct_steps = []
        if self.can_reduce():
            rhs = self.stack[-1]
            lhs = self.stack[-2]
            new_span = range(lhs.span.start, rhs.span.stop)

            for n in gold_tree.iter_nodes():
                if n.span == new_span:
                    correct_steps.append(Step('reduce', n.label, n.direction))
                    break
            else:
                if self.can_shift():
                    correct_steps.append(Step('shift', 'None', 'None'))
                else:
                    # print("all spans:")
                    # for s in [x.span for x in list(gold_tree.iter_nodes())]:
                    #     print("\t", s)
                    raise Exception("There is no correct action given the current state of the parser.")
        elif self.can_shift():
            correct_steps.append(Step('shift', 'None', 'None'))
        return correct_steps

from copy import deepcopy
from collections import namedtuple
from rst import TreeNode

Step = namedtuple('Step', 'action label')

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
            # print("action == shift, line 27 in transition system")
            self.shift()
        elif action == 'reduceL':
            self.reduceL(*args, **kwargs)
        elif action == 'reduceR':
            self.reduceR(*args, **kwargs)

    def shift(self):
        node = self.buffer.pop(0)
        self.stack.append(node)

    def reduce(self, label=None, direction=None, reduce_fn=None):
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        emb = None if reduce_fn is None else reduce_fn(lhs.embedding, rhs.embedding)
        node = TreeNode(children=[lhs, rhs], label=label, direction=direction, embedding=emb)
        node.calc_span()
        self.stack.append(node)


    def reduceL(self, label=None, direction=None, reduce_fn=None):
        self.reduce(label, 'RightToLeft', reduce_fn)
        # rhs = self.stack.pop()
        # lhs = self.stack.pop()
        # emb = None if reduce_fn is None else reduce_fn(lhs.embedding, rhs.embedding)
        # node = TreeNode(children=[lhs, rhs], label=label, direction='RightToLeft', embedding=emb)
        # node.calc_span()
        # self.stack.append(node)

    def reduceR(self, label=None, direction=None, reduce_fn=None):
        self.reduce(label, 'LeftToRight', reduce_fn)
        # rhs = self.stack.pop()
        # lhs = self.stack.pop()
        # emb = None if reduce_fn is None else reduce_fn(lhs.embedding, rhs.embedding)
        # node = TreeNode(children=[lhs, rhs], label=label, direction='LeftToRight', embedding=emb)
        # node.calc_span()
        # self.stack.append(node)

    #todo:
    #keep actions as shift and reduce
    #in take action, call reduce, and pass direction as an arg
    #arcRight = LeftToRight
    


    @staticmethod
    def all_actions():
        return ['shift', 'reduceL', 'reduceR']

    def all_legal_actions(self):
        actions = []
        if self.can_shift():
            actions.append('shift')
        if self.can_reduce():
            actions.append('reduceL')
            actions.append('reduceR')
        # print("all legal actions: ", actions)
        return actions

    def can_shift(self):
        # print("buffer: ", self.buffer)
        # print("len buffer: ", len(self.buffer))
        if len(self.buffer) >= 1:
            # print("can shift")
            return True

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
                # print("gold node: ", n.span)
                if n.span == new_span:
                    # print("n span == new span: ", n.span, " ", new_span)
                    if n.direction == 'LeftToRight':
                        # print(n.direction)
                        correct_steps.append(Step('reduceR', n.label))
                        # break
                    else:
                        correct_steps.append(Step('reduceL', n.label))
                        # print(n.direction)
                    break
                    
            else:
                # print("shift")
                if self.can_shift():
                    correct_steps.append(Step('shift', 'None'))
                else:
                    # print("all spans:")
                    # for s in [x.span for x in list(gold_tree.iter_nodes())]:
                    #     print("\t", s)
                    raise Exception("There is no correct action given the current state of the parser.")
        elif self.can_shift():
            correct_steps.append(Step('shift', 'None'))
        return correct_steps

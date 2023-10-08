import numpy as np


class DecisionTree:
    class Node:
        def __init__(self):
            self.value = None

            self.feature_index = None
            self.children = {}

        def __str__(self):
            if self.children:
                s = 'internal node <%s>:\n' % self.feature_index
                for fv, node in self.children.items():
                    ss = '[%s]-> %s' % (fv, node)
                    s += '\t' + ss.replace('\n', '\n\t') + '\n'
            else:
                s = 'leaf node (%s)' % self.value
            return s

    def __init__(self, gain_threshold=0.12):
        self.gain_threshold = gain_threshold

    def _entropy(self, y):

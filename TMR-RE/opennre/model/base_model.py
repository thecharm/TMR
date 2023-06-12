import torch
from torch import nn
import json

class SentenceRE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def infer(self, item):
        """
        Args:
            item: {'text' or 'token', 'h': {'pos': [start, end]}, 't': ...}
        Return:
            (Name of the relation of the sentence, score)
        """
        raise NotImplementedError
    

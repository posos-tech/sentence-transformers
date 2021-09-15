import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import torch.nn.functional as F
import copy


class ProjectorGivenType(nn.Module):
    """
    Reduce the dim of the tokens with a linear
    """

    def __init__(self,
                 word_embedding_dimension: int,
                 dim_out: int,
                 initialize_with_same_weights: bool = False,
                 freeze_query_one: bool = True
                 ):
        super(ProjectorGivenType, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'dim_out', 'freeze_query_one']

        self.word_embedding_dimension = word_embedding_dimension
        self.dim_out = dim_out

        self.query_projector = nn.Sequential(
            nn.Linear(word_embedding_dimension, word_embedding_dimension),
            nn.ReLU(),
            nn.Linear(word_embedding_dimension, dim_out)
        )
        if initialize_with_same_weights :
            self.doc_projector = copy.deepcopy(self.query_projector)
        else :
            self.doc_projector = nn.Sequential(
                nn.Linear(word_embedding_dimension, word_embedding_dimension),
                nn.ReLU(),
                nn.Linear(word_embedding_dimension, dim_out)
            )

        if freeze_query_one : 
            for param in self.query_projector.parameters():
                param.requires_grad = False

    def get_sentence_embedding_dimension(self):
        return self.dim_out

    def __repr__(self):
        return "SelfAttentionPooler({})".format(self.get_config_dict())

    def forward(self, features):
        emb = features['token_embeddings']
        is_query = features["is_query"]

        if is_query : 
            emb = self.query_projector(emb)
        else :
            emb = self.doc_projector(emb)
        
        features.update({'token_embeddings': emb})
            
        return features
        
        
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = ProjectorGivenType(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))

        return model

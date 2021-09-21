import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import torch.nn.functional as F
import copy


class ProjectorMix(nn.Module):
    """
    Reduce the dim of the tokens with a linear
    """

    def __init__(self,
                 word_embedding_dimension: int,
                 dim_out: int,
                 initialize_with_same_weights: bool = False,
                 differenciate_q_vs_d: bool = False,
                 nb_layers: int = 1,
                 freeze_query_one: bool = True,
                 from_s_emb: bool = False,
                 whiten_params = None
                 ):
        super(ProjectorMix, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'dim_out', 'differenciate_q_vs_d', 'nb_layers', 'freeze_query_one', 'from_s_emb']

        self.word_embedding_dimension = word_embedding_dimension
        self.dim_out = dim_out
        self.initialize_with_same_weights = initialize_with_same_weights
        self.differenciate_q_vs_d = differenciate_q_vs_d
        self.nb_layers = nb_layers
        self.freeze_query_one = freeze_query_one
        self.from_s_emb = from_s_emb

        self.entity_to_pool = "sentence_embedding" if from_s_emb else  "token_embeddings"
        self._initialize_projectors(whiten_params)

    
    def _create_projector(self, whiten_params):
        if self.nb_layers == 1 :
            l = nn.Linear(self.word_embedding_dimension, self.dim_out)
            if whiten_params is not None :
                w_for_model, bias_for_model = whiten_params[0], whiten_params[1].squeeze().dot(whiten_params[0])
                w_for_model = torch.nn.Parameter(torch.tensor(w_for_model.transpose(), dtype=torch.float32))
                bias_for_model = torch.nn.Parameter(torch.tensor(bias_for_model, dtype=torch.float32))
                l.bias = bias_for_model
                l.weight = w_for_model

            return l
        else :
            return nn.Sequential(
                nn.Linear(self.word_embedding_dimension, self.word_embedding_dimension),
                nn.ReLU(),
                nn.Linear(self.word_embedding_dimension, self.dim_out)
            )

    def _initialize_projectors(self, whiten_params) :
        if self.differenciate_q_vs_d :
            self.query_projector = self._create_projector(whiten_params)
            if self.freeze_query_one :
                for param in self.query_projector.parameters():
                    param.requires_grad = False
            if self.initialize_with_same_weights :
                self.doc_projector = copy.deepcopy(self.query_projector)
            else :
                self.doc_projector = self._create_projector(whiten_params)
        else : 
            self.projector = self._create_projector(whiten_params)


    def get_sentence_embedding_dimension(self):
        return self.dim_out

    def __repr__(self):
        return "ProjectorMix({})".format(self.get_config_dict())

    def forward(self, features):
        emb = features[self.entity_to_pool]

        if self.differenciate_q_vs_d :
            is_query = features["is_query"]
            if is_query : 
                emb = self.query_projector(emb)
            else :
                emb = self.doc_projector(emb)
            
        else :
            emb = self.projector(emb)
        
        features.update({self.entity_to_pool: emb})
            
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

        model = ProjectorMix(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))

        return model

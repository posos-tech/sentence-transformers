import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json


class LinearReduction(nn.Module):
    """
    Reduce the dim of the tokens with a linear
    """

    def __init__(self,
                 word_embedding_dimension: int,
                 output_word_dimension: int,
                 normalize: bool = True
                 ):
        super(LinearReduction, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'output_word_dimension', 'normalize']

        self.word_embedding_dimension = word_embedding_dimension
        self.output_word_dimension = output_word_dimension
        self.normalize = normalize

        self.main_linear = nn.Linear(self.word_embedding_dimension, self.output_word_dimension)

    def __repr__(self):
        return "LinearReduction({})".format(self.get_config_dict())

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        token_embeddings = self.main_linear(token_embeddings)
        if self.normalize :
            token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)

        features.update({'token_embeddings': token_embeddings})
        return features

    def get_sentence_embedding_dimension(self):
        return self.output_word_dimension

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

        model = LinearReduction(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))

        return model

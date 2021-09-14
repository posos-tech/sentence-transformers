import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn


class SelfAttentionPooler(nn.Module):
    """
    Reduce the dim of the tokens with a linear
    """

    def __init__(self,
                 word_embedding_dimension: int,
                 dim_out: int
                 ):
        super(SelfAttentionPooler, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'dim_out']

        self.word_embedding_dimension = word_embedding_dimension
        self.dim_out = dim_out


        self.doc_Wq = nn.Linear(self.word_embedding_dimension, self.word_embedding_dimension)
        self.doc_Wk = nn.Linear(self.word_embedding_dimension, self.word_embedding_dimension)
        self.doc_Wv = nn.Linear(self.word_embedding_dimension, self.dim_out)

        self.query_Wq = nn.Linear(self.word_embedding_dimension, self.word_embedding_dimension)
        self.query_Wk = nn.Linear(self.word_embedding_dimension, self.word_embedding_dimension)
        self.query_Wv = nn.Linear(self.word_embedding_dimension, self.dim_out)

        self.attention = ScaledDotProductAttention(temperature=word_embedding_dimension ** 0.5)

    def get_sentence_embedding_dimension(self):
        return self.dim_out

    def __repr__(self):
        return "SelfAttentionPooler({})".format(self.get_config_dict())

    def forward(self, features):
        emb = features['token_embeddings']
        mask = features["attention_mask"]
        is_query = features["is_query"]

        if is_query :
            q, k, v = self.query_Wq(emb), self.query_Wk(emb), self.query_Wv(emb)
        else :
            q, k, v = self.doc_Wq(emb), self.doc_Wk(emb), self.doc_Wv(emb)
        
        mask = mask.unsqueeze(1)

        v, attn = self.attention(q, k, v, mask=mask)

        features.update({"token_embeddings": v})

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

        model = SelfAttentionPooler(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))

        return model

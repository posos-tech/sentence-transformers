import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import torch.nn.functional as F

from sentence_transformers.models.AttentionPooler import ScaledDotProductAttention


class CrossAttentionPooling(nn.Module):
    """
    Pool either the doc or query with an attention of the sentence embedding of the other
    """

    def __init__(self,
                 word_embedding_dimension: int,
                 attention_on_query: bool = True,
                 pooling_strategy_for_the_other: str = "mean"
                 ):
        super(CrossAttentionPooling, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'attention_on_query', 'pooling_strategy_for_the_other']

        self.word_embedding_dimension = word_embedding_dimension
        self.attention_on_query = attention_on_query
        self.pooling_strategy_for_the_other = pooling_strategy_for_the_other

        self.Wq = nn.Linear(self.word_embedding_dimension, self.word_embedding_dimension)
        self.Wk = nn.Linear(self.word_embedding_dimension, self.word_embedding_dimension)
        self.Wv = nn.Linear(self.word_embedding_dimension, self.word_embedding_dimension)

        self.attention = ScaledDotProductAttention(temperature=word_embedding_dimension ** 0.5)

    def _pool_with_strategy(self, emb):
        if self.pooling_strategy_for_the_other == "mean":
            return torch.mean(emb, dim=-1)
        elif self.pooling_strategy_for_the_other == "cls":
            if len(emb.shape) == 3 :
                return emb[:,0,:]
            else :
                return emb[0,:]
        else :
            return torch.max(emb, dim=-1)
    

    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension

    def __repr__(self):
        return "CrossAttentionPooling({})".format(self.get_config_dict())

    def forward(self, features_query, features_doc):
        emb_query = features_query['token_embeddings']
        emb_doc = features_doc['token_embeddings']
        

        if self.attention_on_query :
            attention_emb, query_for_attention = emb_query, emb_doc
            mask_for_attention = features_query["attention_mask"].unsqueeze(1)
        else :
            query_for_attention, attention_emb = emb_query, emb_doc
            mask_for_attention = features_doc["attention_mask"].unsqueeze(1)
        
        query_for_attention = self._pool_with_strategy(query_for_attention)
        q = self.Wq(query_for_attention)

        k = self.Wk(attention_emb)
        v = self.Wk(attention_emb)
    
        v, attn = self.attention(q, k, v, mask=mask_for_attention)

        return v, query_for_attention
        
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

        model = CrossAttentionPooling(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))

        return model

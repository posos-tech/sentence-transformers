import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer, util
from .. import InputExample
import numpy as np


def sim_colbert(Q, D):
    cosines = (Q @ D.permute(0, 2, 1)).max(1).values.sum(1)
    return cosines


class ColbertLoss(nn.Module):
    ''' 
    should be triples (query, Doc_pos, Doc_neg)
    the loss will be the cross entropy of the two scores to maximize
    '''
    def __init__(self, model: SentenceTransformer):
        """
        :param model: SentenceTransformer model
        """
        super(ColbertLoss, self).__init__()
        # This will be the final model used during the inference time.
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, triples: Iterable[Dict[str, Tensor]], labels: Tensor):
        q_features, dpos_features, dneg_features = tuple(triples)
        # (bsz, n_token, hdim)
        q_features = self.model(q_features)
        dpos_features = self.model(dpos_features)
        dneg_features = self.model(dneg_features)

        q_features_post_treated = self.post_treat_emb(q_features["token_embeddings"], q_features["attention_mask"])
        dpos_features_post_treated = self.post_treat_emb(dpos_features["token_embeddings"], dpos_features["attention_mask"])
        dneg_features_post_treated = self.post_treat_emb(dneg_features["token_embeddings"], dneg_features["attention_mask"])

        cosines_pos = sim_colbert(q_features_post_treated, dpos_features_post_treated)
        cosines_neg = sim_colbert(q_features_post_treated, dneg_features_post_treated)

        scores = torch.stack((cosines_pos, cosines_neg), dim=1)
        labels = torch.zeros(len(scores), dtype=torch.long, device=scores.device)

        loss = self.criterion(scores, labels)
        return loss        

    def post_treat_emb(self, emb, mask):
        input_mask_expanded = mask.unsqueeze(
                    -1).expand(emb.size()).float()
        return emb*input_mask_expanded
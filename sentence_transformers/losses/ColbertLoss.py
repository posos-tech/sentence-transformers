import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer, util
from .. import InputExample
import numpy as np
import torch.nn.functional as F
from BertBase.SentenceTransfo.Supervised.utils_colbert import *
    

class ColbertLoss(nn.Module):
    ''' 
    should be triples (query, Doc_pos, Doc_neg)
    the loss will be the cross entropy of the two scores to maximize
    '''
    def __init__(self, model: SentenceTransformer, use_sentence_embeddings=False, use_cross_entropy=True, margin=0.5):
        """
        :param model: SentenceTransformer model
        """
        super(ColbertLoss, self).__init__()
        # This will be the final model used during the inference time.
        self.model = model
        self.use_sentence_embeddings = use_sentence_embeddings
        self.use_cross_entropy = use_cross_entropy
        if self.use_cross_entropy :
            self.criterion = nn.CrossEntropyLoss()
        else :
            self.criterion = nn.MarginRankingLoss(margin=margin)

    def forward(self, triples: Iterable[Dict[str, Tensor]], labels: Tensor):
        q_features, dpos_features, dneg_features = tuple(triples)
        # (bsz, n_token, hdim)

        q_features = self.model(q_features)
        dpos_features = self.model(dpos_features)
        dneg_features = self.model(dneg_features)

        '''
        if self.use_sentence_embeddings :
            print(len(q_features["sentence_embedding"]))
            print(q_features["sentence_embedding"][0].shape)
            cosines_pos = F.cosine_similarity(q_features["sentence_embedding"], dpos_features["sentence_embedding"])
            cosines_neg = F.cosine_similarity(q_features["sentence_embedding"], dneg_features["sentence_embedding"])
        '''
        #else :
        q_features_post_treated = self.post_treat_emb(q_features["token_embeddings"], q_features["attention_mask"])
        dpos_features_post_treated = self.post_treat_emb(dpos_features["token_embeddings"], dpos_features["attention_mask"])
        dneg_features_post_treated = self.post_treat_emb(dneg_features["token_embeddings"], dneg_features["attention_mask"])

        cosines_pos = sim_colbert(q_features_post_treated, dpos_features_post_treated)
        cosines_neg = sim_colbert(q_features_post_treated, dneg_features_post_treated)

        if self.use_cross_entropy :
            scores = torch.stack((cosines_pos, cosines_neg), dim=1)
            labels = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
            loss = self.criterion(scores, labels)
        
        else : 
            y = torch.zeros(len(cosines_pos), dtype=torch.long, device=cosines_pos.device)+1
            self.criterion(cosines_pos, cosines_neg, y)

        return loss        

    def post_treat_emb(self, emb, mask):
        input_mask_expanded = mask.unsqueeze(
                    -1).expand(emb.size()).float()
        return emb*input_mask_expanded

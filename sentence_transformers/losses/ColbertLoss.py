import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer, util
from torch.nn.modules import loss
from .. import InputExample
import numpy as np
import torch.nn.functional as F
from BertBase.SentenceTransfo.Supervised.utils_colbert import *
    

def cosine_custom(a, b):
    n_a, n_b = len(a.shape), len(b.shape)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)
    if n_b == 3 and n_a==2:
        a_norm = a_norm.unsqueeze(1)
        return (a_norm @ b_norm.permute(0, 2, 1)).squeeze()
    elif n_b==2 and n_a==2:
        return (a_norm*b_norm).sum(1)
    elif n_b==1 and n_a == 1:
        return torch.dot(a_norm, b_norm)
    else :
        raise "not implemented yet"

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
            loss = self.criterion(cosines_pos, cosines_neg, y)

        return loss        

    def post_treat_emb(self, emb, mask):
        input_mask_expanded = mask.unsqueeze(
                    -1).expand(emb.size()).float()
        return emb*input_mask_expanded


class ColbertMultipleNegsLoss(nn.Module):
    ''' 
    should be triples (query, Doc_pos, Doc_neg)
    the loss will be the cross entropy of the two scores to maximize
    '''
    def __init__(self, model: SentenceTransformer, use_cross_entropy=True, margin=0.5, is_sentence_embeddings=True):
        """
        :param model: SentenceTransformer model
        """
        super(ColbertMultipleNegsLoss, self).__init__()
        # This will be the final model used during the inference time.
        self.model = model
        self.use_cross_entropy = use_cross_entropy

        self.is_sentence_embeddings = is_sentence_embeddings

        if self.use_cross_entropy :
            self.criterion = nn.CrossEntropyLoss()
        else :
            self.criterion = nn.TripletMarginLoss(margin=margin)


    def forward(self, triples: Iterable[Dict[str, Tensor]], labels: Tensor):
        q_features, dpos_features, *dneg_features = tuple(triples)
        
        if self.is_sentence_embeddings :
            loss = self.forward_sentence_embedding_style(q_features, dpos_features, dneg_features)
        else :
            loss = self.forward_colbert_style(q_features, dpos_features, dneg_features)

        return loss

    def forward_sentence_embedding_style(self, q_features, dpos_features, dneg_features):
        q_features = self.model(q_features)
        dpos_features = self.model(dpos_features)

        dneg_stacked = torch.stack([self.model(dn)["sentence_embedding"] for dn in dneg_features]).transpose(0,1)

        cosine_pos = cosine_custom(q_features["sentence_embedding"], dpos_features["sentence_embedding"])
        cosine_neg = cosine_custom(q_features["sentence_embedding"], dneg_stacked)

        scores = torch.cat((cosine_pos.unsqueeze(1), cosine_neg), dim=1)
        labels = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
        loss = self.criterion(scores, labels)

        return loss

    def forward_colbert_style(self, q_features, dpos_features, dneg_features):
        q_features = self.model(q_features)
        dpos_features = self.model(dpos_features)

        q_features_post_treated = self.post_treat_emb(q_features["token_embeddings"], q_features["attention_mask"])
        dpos_features_post_treated = self.post_treat_emb(dpos_features["token_embeddings"], dpos_features["attention_mask"])

        cosines_negs = []
        for i, dn in  enumerate(dneg_features) :
            dn = self.model(dn)
            dn = self.post_treat_emb(dn["token_embeddings"], dn["attention_mask"])
            cosines_negs.append(sim_colbert(q_features_post_treated, dn))

        cosines_pos = sim_colbert(q_features_post_treated, dpos_features_post_treated)

        if self.use_cross_entropy :
            scores = torch.stack([cosines_pos]+ cosines_negs, dim=1)
            labels = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
            loss = self.criterion(scores, labels)
        
        else : 
            y = torch.zeros(len(cosines_pos), dtype=torch.long, device=cosines_pos.device)+1
            loss = self.criterion(cosines_pos, cosines_negs, y)

        return loss        

    def post_treat_emb(self, emb, mask):
        input_mask_expanded = mask.unsqueeze(
                    -1).expand(emb.size()).float()
        return emb*input_mask_expanded

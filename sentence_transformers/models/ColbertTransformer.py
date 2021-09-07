from sentence_transformers import models
import tqdm
import numpy as np
import torch
import os
from typing import List, Dict, Optional, Union, Tuple 
import json
import string

from BertBase.SentenceTransfo.Whitening_helpers.whithen_utils import compute_kernel_bias, reduce_kernel_dimension



class ColbertTransformer(models.Transformer):
    def __init__(self, query_maxlen, doc_maxlen, input_path, filter_punctuation=False):
        # Loading previous model Bert
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)

        if "max_seq_length" in config :
            config["max_seq_length"] = max(doc_maxlen, query_maxlen)

        super(ColbertTransformer, self).__init__(input_path, **config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen

        self.config_keys = ['max_seq_length', 'query_maxlen', 'doc_maxlen', 'do_lower_case']

        self.filter_punctuation = filter_punctuation
        self.instantiate_for_tokenizer()
        


    def instantiate_for_tokenizer(self):
        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tokenizer.convert_tokens_to_ids('[unused0]')
        self.D_marker_token, self.D_marker_token_id = '[D]', self.tokenizer.convert_tokens_to_ids('[unused1]')

        self.cls_token, self.cls_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.sep_token, self.sep_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.mask_token, self.mask_token_id = self.tokenizer.mask_token, self.tokenizer.mask_token_id

        if self.filter_punctuation :
            list_tokens = []
            punctuations = string.punctuation
            for p in punctuations:
                list_tokens.append(self.tokenizer.convert_tokens_to_ids(p))

        self.punctuation_tokens = torch.tensor(list_tokens)


    def __repr__(self):
        # super(Transformer, self).__repr__()
        return "ColbertTransformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)

    # overide the tokenizer method
    def tokenize(self, texts: Union[List[str]], is_query):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        to_tokenize = [['. ' + s for s in col] for col in to_tokenize]

        if is_query is not None :
            out_tok = self.tokenize_given_type(to_tokenize, is_query)
            output.update(out_tok)
        else :
            for i, col in enumerate(to_tokenize) :
                output.update(self.tokenize_given_type(col, i==0))

        return output
    
    def tokenize_given_type(self, to_tokenize, is_query):
        obj = self.tokenizer(*to_tokenize, padding='max_length' if is_query else 'longest', truncation=True if is_query else 'longest_first',
                    return_tensors='pt', max_length=self.query_maxlen if is_query else self.doc_maxlen)

        ids = obj['input_ids']

        if is_query :
            ids[:, 1] = self.Q_marker_token_id
            ids[ids == 0] = self.mask_token_id
            obj['attention_mask'][obj['attention_mask']==0] = 1
        else :
            ids[:, 1] = self.D_marker_token_id

        obj['attention_mask'][:,:2] = 0   # we won't compute maxsim from the two first tokens
        obj['input_ids'] = ids

        if self.filter_punctuation:
            for id_punct in self.punctuation_tokens:
                obj['attention_mask'][ids==id_punct] = 0
        
        return obj








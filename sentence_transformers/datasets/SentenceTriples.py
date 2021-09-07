from torch.utils.data import Dataset
from typing import List
from ..readers.InputExample import InputExample
from tqdm import tqdm


class SentenceTriplesDataset(Dataset):
    """
    The SentenceTriplesDataset returns InputExamples in the format: texts=[sentence_1, sentence_close, sentence_far]
    It should be used with a marginal loss

    :param sentences: A list of sentences
    """
    def __init__(self, sentences: List[str], parser=None):
        self.parser = parser

        # Create the triples if a parser is passed and thus the triples are not already made
        if self.parser is not None :
            triples = []
            for s in tqdm(sentences) :
                s_close, s_far = self.parser(s)
                if not len(s_close)==0 :
                    triples.append((s, s_close, s_far))
        else :
            triples = sentences

        self.triples = triples

    def __getitem__(self, item):
        sent, sent_close, sent_far = self.triples[item]
        return InputExample(texts=[sent, sent_close, sent_far])

    def __len__(self):
        return len(self.triples)

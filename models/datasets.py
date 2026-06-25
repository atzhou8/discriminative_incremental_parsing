import torch
import pandas as pd

from pathlib import Path

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from stanza.utils.conll import CoNLL


class TreebankDataset(Dataset):
    """Dataset for loading sentences and gold trees 
    from a conllu file.
    """

    def __init__(self, data_dir):
        super(TreebankDataset, self).__init__()
        self.trees = CoNLL.conll2doc(data_dir).sentences

    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx):
        """Gets sentence as list of nodes in tree and tree structure as a 
        list of heads."""
        # initial 0 for distinguished root
        heads = []
        words = []
        is_adjunct = []
        for word in self.trees[idx].words:
            heads.append(word.head)
            words.append(word.text)
            is_adjunct.append(1 if word.deprel.startswith('mod') else 0)
            
        is_adjunct = torch.Tensor([0] + is_adjunct) 
        tree = torch.Tensor([0] + heads) # 0 for root
        words = [w.text for w in self.trees[idx].words]
        length = len(tree)

        return words, tree, length, is_adjunct

def treebank_collater(batch, cutoff_transform=None):
    if cutoff_transform is None:
        cutoff_transform = lambda x: x
    sentences, trees, lengths, is_adjunct = zip(*batch)
    trees = pad_sequence(
        list(trees), 
        batch_first=True, 
        padding_value=0
    ).long()
    is_adjunct = pad_sequence(
        list(is_adjunct), 
        batch_first=True, 
        padding_value=0
    ).long()
    lengths = torch.tensor(list(lengths), dtype=torch.int64)
    
    return {
        'sentences': list(sentences), 
        'gold_trees': trees, 
        'lengths': lengths,
        'cutoffs': cutoff_transform(None),
        'conditions': None,
        'gold_adjuncts': is_adjunct
    }

class PhenomenaDataset(Dataset):
    """Dataset for loading in sentences and disambiguation points for 
    the purposes of psycholinguistic evaluation"""

    def __init__(
            self, 
            data_dir,
            condition_col='condition',
            pos_col='critical_pos',
            sentence_col='sentence',
    ):
        super(PhenomenaDataset, self).__init__()
        df = pd.read_csv(data_dir)
        self.conditions = df['condition'].tolist()
        self.cutoffs = df['critical_pos'].tolist()
        self.sentences = df['sentence'].tolist()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx].split(' ') # punct is presplit in file
        length = len(words) + 1 # add 1 for distinguished root node
        cutoff = self.cutoffs[idx]
        condition = self.conditions[idx]
       
        return words, length, cutoff, condition
    
def phenomena_collater(batch, cutoff_transform=None):
    if cutoff_transform is None:
        cutoff_transform = lambda x: x
    sentences, lengths, cutoffs, conditions = zip(*batch)
    lengths = torch.tensor(list(lengths), dtype=torch.int64)
    cutoffs = torch.tensor(list(cutoffs), dtype=torch.int64)
    return {
        'sentences': list(sentences), 
        'gold_trees': None, 
        'lengths': lengths,
        'cutoffs': cutoff_transform(cutoffs),
        'conditions': conditions,
        'gold_adjuncts': None
    }

class SynSurpDataset(Dataset):

    def __init__(self, sentence_dir, tag_dir):
        self.all_tags = Path(tag_dir).read_text(encoding='utf-8').splitlines()
        self.all_sentences = Path(sentence_dir).read_text(encoding='utf-8').splitlines()

        assert len(self.all_tags) == len(self.all_sentences)

    def __len__(self):
        return len(self.all_tags)
    
    def __getitem__(self, idx):
        tags = self.all_tags[idx].split(' ')
        words = self.all_sentences[idx].split(' ')
        assert(len(tags) == len(words))
        return words, tags

def synsurp_collator(batch):
    sentences, tags = zip(*batch)
    return {
        'sentences': sentences,
        'tags': tags
    }

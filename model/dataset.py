import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from stanza.utils.conll import CoNLL

class ParsingDataset(Dataset):
    """Dataset for loading sentence embeddings and gold trees 
    from a conllu file.
    """

    def __init__(self, data_dir):
        super(ParsingDataset, self).__init__()
        self.trees = CoNLL.conll2doc(data_dir).sentences

    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx):
        """Gets sentence as list of nodes in tree and tree structure as a 
        list of heads."""
        # initial 0 for distinguished root
        tree = torch.Tensor([0] + [w.head for w in self.trees[idx].words])
        words = [w.text for w in self.trees[idx].words]
        length = len(tree)

        return words, tree, length

def parsing_collater(batch):
    sentences, trees, lengths = zip(*batch)
    trees = pad_sequence(list(trees), batch_first=True, padding_value=0).long()
    lengths = torch.tensor(list(lengths), dtype=torch.int64)
    
    return list(sentences), trees, lengths

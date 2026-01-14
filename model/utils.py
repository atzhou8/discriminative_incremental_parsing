import torch

from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from torch.utils.data import DataLoader

from .dataset import ParsingDataset, parsing_collater


def tensors_to_conllu(words, heads, write_path):
    if type(heads) is torch.Tensor:
        heads = heads.detach().cpu().tolist()

    batch_size = len(words)
    sentences = []
    for b in range(batch_size):
        sentence = []
        for idx in range(len(words[b])): 
            word = words[b][idx]
            head = heads[b][idx+1]
            sentence.append({
                'id': idx+1,
                'text': word,
                'head': head,
            })
        sentences.append(sentence)

    doc = Document(sentences)
    CoNLL.write_doc2conll(doc, write_path)

    return doc

def build_loader(path, batch_size, shuffle=False, num_workers=6):
    dataset = ParsingDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=parsing_collater,
        num_workers=num_workers,
        pin_memory=True,
    )
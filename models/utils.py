import torch
import numpy as np

from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter

from .datasets import *

def get_vocab_from_text(file_name, min_count=1):
    tag_seqs = Path(file_name).read_text(encoding='utf-8').splitlines()
    counter = Counter()
    for seq in tag_seqs:
        if not seq:
            continue
        counter.update(seq.split())

    tags = [tag for tag, cnt in counter.items() if cnt >= min_count]
    tags.extend(['<bos>', '<eos>', '<oov>'])

    return sorted(tags)

def tensors_to_conllu(words, heads, write_path):
    if type(heads) is torch.Tensor:
        heads = heads.detach().cpu().tolist()

    write_path = Path(write_path)
    write_path.parent.mkdir(parents=True, exist_ok=True)

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

def build_loader(
        path, 
        batch_size, 
        shuffle=False, 
        num_workers=0, 
        dataset_type='treebank',
        cutoff_fn = None
):
    assert dataset_type in ['treebank', 'phenomena'], \
          'Dataset type must be either treebank or phenomena'
    D = TreebankDataset if dataset_type=='treebank' else PhenomenaDataset
    col_fn = (lambda x: treebank_collater(x, cutoff_fn)) if dataset_type=='treebank' \
             else (lambda x: phenomena_collater(x, cutoff_fn))

    return DataLoader(
        D(path),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=col_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

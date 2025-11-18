import torch

from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document

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

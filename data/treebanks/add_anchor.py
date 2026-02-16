import argparse
import copy

from tqdm import tqdm
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir')
parser.add_argument('-o', '--output_dir', default=None)


if __name__ == '__main__':
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.input_dir.split('.')[0]
        output_dir += '_anchor.conllu'
    
    full_trees = CoNLL.conll2doc(input_dir).sentences
    output_trees = []

    print("Adding anchor nodes to trees...")
    for full_tree in tqdm(full_trees):
        full_tree = full_tree.to_dict()
        full_tree = [
            token for token in full_tree if isinstance(token['id'], int)
        ]

        # Shift all existing token ids by 1
        for token in full_tree:
            token['id'] += 1
            if token['head'] != 0:
                token['head'] += 1
            token['deps'] = None

        # Insert anchor node as the first token (id=1, head=0, no dependents)
        anchor = {
            'id': 1,
            'text': '<anchor>',
            'lemma': '<anchor>',
            'upos': '_',
            'xpos': '_',
            'feats': '_',
            'head': 0,
            'deprel': '_',
            'deps': None,
            'misc': '_',
        }

        new_tree = [anchor] + full_tree
        output_trees.append(new_tree)

    print("Saving trees as Stanza Document...")
    doc = Document(output_trees)
    print("Writing...")
    CoNLL.write_doc2conll(doc, output_dir)
    print("Done.")
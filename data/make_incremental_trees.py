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
        output_dir += '_incremental.conllu'
    
    full_trees = CoNLL.conll2doc(input_dir).sentences
    partial_trees = []

    print("Converting trees...")
    for full_tree in tqdm(full_trees):
        full_tree = full_tree.to_dict()
        full_tree = [
            token for token in full_tree if isinstance(token['id'], int)
        ]
        num_words = len(full_tree)
        for prefix_end in range(2, num_words):
            prefix = copy.deepcopy(full_tree[0:prefix_end])
            # Set all floating nodes as roots
            for node in prefix:
                if node['head'] > prefix_end:
                    node['head'] = 0
                node['deps'] = None

            # Save original prefix then a copy with NEXT
            partial_trees.append(prefix)

    print("Saving trees as Stanza Document...")
    doc = Document(partial_trees)
    print("Writing...")
    CoNLL.write_doc2conll(doc, output_dir)
    print("Done.")


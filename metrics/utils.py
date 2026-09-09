import stanza
import torch
import re

tokenizer = None


def _load_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = stanza.Pipeline(
            lang='en',
            processors='tokenize,mwt',
            use_gpu=torch.cuda.is_available()
        )

def get_prefix_prompts_by_token(sentence):
    _load_tokenizer()
    sent = tokenizer(sentence).sentences[0]

    prefixes = []
    words_so_far = []
    for tok in sent.tokens:
        if len(tok.words) == 1: # full word
            words_so_far.append(tok.words[0].text)
            hyphen_end = tok.start_char
            for part in tok.text.split('-')[:-1]: # compound special rule
                hyphen_end += len(part) + 1
                prefixes.append((list(words_so_far), sentence[:hyphen_end]))
            prefixes.append((list(words_so_far), sentence[:tok.end_char]))
        else: # has tokens
            surface = tok.text
            pos = 0
            for w in tok.words:
                idx = surface.lower().find(w.text.lower(), pos)
                end = (idx + len(w.text)) if idx != -1 else pos
                pos = end
                words_so_far.append(w.text)
                prefixes.append((list(words_so_far), sentence[:tok.start_char + end]))

    return prefixes[:-1]

def split_sentence_by_token(text):
    """Tokenization scheme for dependency trees"""
    _load_tokenizer()
    items = tokenizer(text).sentences[0].tokens # type: ignore
    return [w.text for tok in items for w in tok.words]

def split_sentence_by_word(text):
    """Tokenization scheme for eye-tracking and SPR"""
    pattern = r'\s+|(?<=-)'
    return [word for word in re.split(pattern, text) if word]

def create_word_rows_for_sentence(sentence):
    words = split_sentence_by_word(sentence)

    word_rows = []
    for word in words:
        word_rows.append({
            'unsplit': word,
            'split': split_sentence_by_token(word)
        })
    return word_rows

def expand_sentences_to_word_rows(items_df):
    """Expands df with a sentence per row to instead contain one token per row."""
    base_columns = list(items_df.columns)
    new_columns = [
        'SentenceTokenized',
        'word_pos', 
        'EachWord',
        'WordTokens',
        'SentenceStart',
        'WordStart', 
        'IsMultiWord',
        'WordStart',
    ]
    output_dict = {column: [] for column in base_columns + new_columns}

    for idx, (_, row) in enumerate(items_df.iterrows()):
        sentence_word_rows = create_word_rows_for_sentence(row['Sentence'])
        tokenized_sentence = [item for word_row in sentence_word_rows for item in word_row['split'] ]
        pos = 1
        for i, word_row in enumerate(sentence_word_rows):
            each_word = word_row['unsplit']
            word_tokens = word_row['split']
            for j, token in enumerate(word_tokens):
                for column in base_columns:
                    output_dict[column].append(row[column])
                output_dict['SentenceTokenized'].append(tokenized_sentence)
                output_dict['word_pos'].append(pos)
                output_dict['EachWord'].append(each_word)
                output_dict['WordTokens'].append(token)
                output_dict['SentenceStart'].append(i==0 and j==0)
                output_dict['WordStart'].append(j==0)
                output_dict['IsMultiWord'].append(len(word_tokens)>1)
                pos += 1

    return output_dict

def combine_multi_words(word_rows, metrics_to_save):
    """After writing metrics by token, sum metric values over tokenized words 
    and delete non-initial tokens"""
    indices_to_drop = []
    deleted_in_curr_sent = 0
    for word_index, row in word_rows.iterrows():
        if row['SentenceStart']:
            deleted_in_curr_sent = 0
        word_rows.loc[word_index, 'word_pos'] -= deleted_in_curr_sent
        if row['IsMultiWord'] and row['WordStart']:
            found_full_word = False
            increment = 1
            word_length = 1
            while not found_full_word and word_index+increment < len(word_rows):
                next_row = word_rows.iloc[word_index+increment]
                if next_row['WordStart']:
                    found_full_word = True
                else:
                    indices_to_drop.append(word_index+increment)
                    deleted_in_curr_sent += 1
                    word_length += 1
                    increment += 1 
                    for metric in metrics_to_save:
                        word_rows.loc[word_index, metric] += next_row[metric]


            
            for metric in metrics_to_save:
                word_rows.loc[word_index, metric] /= 1

    word_rows.drop(index=indices_to_drop, inplace=True)
    word_rows.reset_index(drop=True, inplace=True)

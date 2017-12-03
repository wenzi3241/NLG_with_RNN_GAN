import helper
import numpy as np
import problem_unittests as tests
from collections import Counter

data_dir = './data/simpsons/sheldon.txt'
#data_dir = './data/simpsons/moes_tavern_lines.txt'

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return(vocab_to_int, int_to_vocab)

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    tokenized_text = {
        '.':'<PERIOD>',
        ',':'<COMMA>',
        '"':'<QUOTATION_MARK>',
        ';':'<SEMICOLON>',
        '!':'<EXCLAMATION_MARK>',
        '?':'<QUESTION_MARK>',
        '(':'<LEFT_PAREN>',
        ')':'<RIGHT_PAREN>',
        '--':'<DASH>',
        '\n':'<RETURN>'
    }
    
    return tokenized_text

helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

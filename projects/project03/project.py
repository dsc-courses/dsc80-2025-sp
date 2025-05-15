# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    time.sleep(0.5) 
    response = requests.get('https://www.gutenberg.org/ebooks/16328.txt.utf-8')
    book_text = response.text

    start_marker = '*** START OF'
    end_marker = '*** END OF'

    start_ix = book_text.find(start_marker)
    end_ix = book_text.find(end_marker)

    # skip start 
    skip_start = book_text.find('\n', start_ix)
        
    # Slice between markers (skip marker itself)
    content = book_text[skip_start:end_ix]

    content = content.replace('\r\n', '\n')

    return content


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    # split our string by potential paragraphs 
    # do not add paragraph start and stop indicators yet 
    para_marked = re.split(r'\n{2,}', book_string)

    # remove any possible white spaces 
    para_marked = filter(lambda p: p.strip(), para_marked)

    # now we add paragraph start and stop indicators 
    add_para_markers = map(lambda p: '\x02' + p + '\x03', para_marked)
    add_para_markers = list(add_para_markers)

    # combine back into one string to use regex functions again 
    full_str_w_markers = ' '.join(add_para_markers)
    full_str_w_markers

    # 1st) capture underscores as punctuation sandwiched within a word, control where underscores appear 
        # \w are alphanumerics including underscores 
        # ?: non capture group, matches substring but does not store it as a group
        # for example w/o '?:' we are saying 'capture groups that have underscore followed by a word' 
        # aka ignore dangling/stray underscores 
    # 2nd) separate punctuations using 'or'
        # [...] matching groups 
        # ^ not an alphanumeric or white space 
    # 3rd) separate our paragraph markers using 'or' 
    token_pattern = r"\w+(?:_\w)*|[^\w\s]|\x02|\x03"

    tokenized = re.findall(token_pattern, full_str_w_markers)

    return tokenized

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        uniq_idx = pd.Series(tokens).unique()
        total_omega = len(uniq_idx)
        self.mdl = pd.Series(data= 1/total_omega,index=uniq_idx)

        return self.mdl
    
    def probability(self, words):
        if pd.Series(words).isin(self.mdl.index).sum() != len(words):
            return 0
        
        probability = self.mdl.iloc[1] ** len(words)

        return probability
        
    def sample(self, M):
        rand_tokens = np.random.choice(self.mdl.index, size=M, p=self.mdl.values)

        return ' '.join(rand_tokens)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        intermediary_ser = pd.Series(tokens).value_counts()

        # ||omega|| is the total number of tokens not just the number of unique tokens 
        self.mdl = pd.Series(data = intermediary_ser.values/len(tokens), index = intermediary_ser.index)

        return self.mdl
    
    def probability(self, words):
        if pd.Series(words).isin(self.mdl.index).sum() != len(words):
            return 0
            
        words_ser = pd.Series(words).value_counts() 
        probability = self.mdl.loc[words_ser.index] ** words_ser.values
        
        return probability.prod()
        
    def sample(self, M):
        rand_tokens = np.random.choice(self.mdl.index, size=M, p=self.mdl.values)

        return ' '.join(rand_tokens)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        ...
        
    def train(self, ngrams):
        ...
    
    def probability(self, words):
        ...
    

    def sample(self, M):
        ...

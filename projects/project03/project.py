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
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        ...
    
    def probability(self, words):
        ...
        
    def sample(self, M):
        ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        ...
    
    def probability(self, words):
        ...
        
    def sample(self, M):
        ...


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

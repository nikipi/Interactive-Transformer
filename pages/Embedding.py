import streamlit as st
import numpy as np
import pandas as pd


# @st.cache

st.markdown("## Embedding Layer")
st.write('''
First of all we need to convert word(or tokens ids ) in the input sequence to an embedding vector. I try to think Embedding vectors as a vector that try to capture the semantic representation of each word.

Suppoese each embedding vector is of 512 dimension and suppose our vocab size is 100, then our embedding matrix will be of size 100x512. These marix will be learned on training and during inference each word will be mapped to corresponding 512 d vector. Suppose we have batch size of 32 and sequence length of 10(10 words). The the output will be 32x10x512.
''')

code = '''

import torch.nn as nn
import math as m 
import torch

class Embeddings(nn.Module):
    def __init__(self, vocab_size, padding_idx, d_model):
        super().__init__()
        self.d_model = d_model 
        self.embed = nn.Embedding(vocab_size,  d_model, padding_idx = padding_idx)

    def forward(self, x):
        embedding = self.embed(x)

        return embedding * m.sqrt(self.d_model)



'''

st.code(code, language='python')
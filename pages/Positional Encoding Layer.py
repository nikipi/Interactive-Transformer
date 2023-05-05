import streamlit as st
import numpy as np
import pandas as pd


# @st.cache

st.markdown("## Positional Encoding Layer")

st.write('''

Positional Encoding will not change the dimension of embedding but it adds information about: what is the position of the word in the sentence.(model then can learn things like the closer words are relevant to each other.)

On odd time steps a cosine function is used and in even time steps a sine function is used.


''')

code = '''
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout = 0.3, max_seq_len = 200, device = 'cpu'):
        super().__init__()
        self.embed_dim = embed_dim 
        self.dropout = nn.Dropout(dropout)
        ### same shape as embedding layer 
        pe = torch.zeros(max_seq_len, embed_dim).to(device)
        ## pos --- the order in the sentence 
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()

        two_i = torch.arange(0, embed_dim, step=2).float()
        denominator = torch.pow(10000, two_i)


        pe[:, 0::2] = torch.sin(pos/ denominator)
        pe[:, 1::2] = torch.cos(pos/ denominator)

        pe = pe.unsqueeze(0)

        # assigns the first argument to a class variable
        # i.e. self.pe
        self.register_buffer("pe", pe)

    def forward(self, x):
        # shape(x) = [B x seq_len x D]
        one_batch_pe: torch.Tensor = self.pe[:, :x.shape[1]].detach()
        repeated_pe = one_batch_pe.repeat([x.shape[0], 1, 1]).detach()
        x = x.add(repeated_pe)
        # shape(x) = [B x seq_len x D]
        return self.dropout(x)


'''

st.code(code, language='python')
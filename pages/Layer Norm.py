import streamlit as st
import numpy as np
import pandas as pd


# @st.cache

st.markdown("## Layer Norm")
st.write('''
Input values in all neurons in the same layer are normalized for each data sample

One fundamental thing about Transformer is: Residual stream

Outputs from previous layers pass to new layers to have new output, and sum up the new output with outputs from previous layers.

''')

code = '''



class ResidualLayerNorm(nn.Module):
    def __init__(self, embed_dim, dropout= 0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        ln = self.layer_norm(self.dropout(x) + residual )

        return ln 

'''

st.code(code, language='python')
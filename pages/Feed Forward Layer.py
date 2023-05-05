import streamlit as st
import numpy as np
import pandas as pd


# @st.cache

st.markdown("## Feed Forward Layer")
st.write('''
Just a standard neural network layer (a hidden layer and an activation layer)
''')

code = '''


class PWFFN(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout = 0.3):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim)
        )

    def forward(self, x):
        ff = self.ff(x)

        return  ff



'''

st.code(code, language='python')
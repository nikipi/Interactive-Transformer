import streamlit as st
import numpy as np
import pandas as pd


# @st.cache

st.markdown("## Encoder")

code = '''



class Encoderlayer(nn.Module):
    def __init__(self, d_ff, embed_dim, num_heads, dropout = 0.3): 
        super().__init__()


        self.norm_1 = ResidualLayerNorm(embed_dim, dropout)
        self.norm_2 = ResidualLayerNorm(embed_dim, dropout)

        self.mha = Mha(embed_dim, num_heads, dropout)

        self.ff = PWFFN(embed_dim, d_ff, dropout)



    def forward(self, x, mask=None):
        # shape(x) = [batch seq_len embed_dim]

        mha, encoder_attn_weights = self.mha(x,x,x,mask=None)

        norm1 = self.norm_1(mha, x)

        ff = self.ff(norm1)

        norm2 = self.norm_2(ff, norm1)

        return norm2, encoder_attn_weights





'''

st.code(code, language='python')


code = '''



class Encoder(nn.Module):
    def __init__(self, Embeddings, embed_dim,
                 num_heads, num_layers,
                 d_ff, device="cpu", dropout=0.3):
        super().__init__()

        self.embedding = Embeddings

        self.PE = PositionalEncoding(
            embed_dim, device=device)

        self.encoders = nn.ModuleList(

            [ Encoderlayer(
                d_ff, 
                embed_dim,
                num_heads, 
                dropout = 0.3
            ) for layer in range(num_layers)]
        )


    def forward(self, x, mask = None):

        embeddings = self.embedding(x)
        encoding = self.PE(embeddings)

        for encoder in self.encoders:
            encoding, encoder_attention_weights = encoder(encoding,mask)

        return  encoding, encoder_attention_weights



'''

st.code(code, language='python')




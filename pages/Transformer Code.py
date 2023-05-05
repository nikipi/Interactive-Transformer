import streamlit as st
import numpy as np
import pandas as pd


# @st.cache

st.markdown("## Transformer")

code = '''



class Transformer(nn.Module):
    def __init__(self, src_vocab_len, trg_vocab_len, embed_dim, d_ff,
                 num_layers, num_heads, src_pad_idx= None , trg_pad_idx= None, dropout=0.3, device="cpu"):
        super().__init__()

        self.num_heads = num_heads
        self.device = device

        encoder_Embedding = Embeddings(src_vocab_len, embed_dim, src_pad_idx)
        decoder_Embedding = Embeddings(trg_vocab_len, embed_dim, trg_pad_idx)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(encoder_Embedding, embed_dim,
                               num_heads, num_layers, d_ff, device, dropout)
        self.decoder = Decoder(decoder_Embedding, embed_dim,
                               num_heads, num_layers, d_ff, device, dropout)

        self.linear_layer = nn.Linear(embed_dim, trg_vocab_len)


    def create_src_mask(self, src): 
            if self.src_pad_idx:
              src_mask = (src != self.src_pad_idx).unsqueeze(1)

              return src_mask 
            
            

    def create_trg_mask(self, trg):
            
             batch_size, trg_len = trg.shape
       
             trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
             batch_size, 1, trg_len, trg_len)
        

             return trg_mask

    def forward(self, src, trg):

        src_mask = self.create_src_mask(src)
        trg_mask = self.create_trg_mask(trg)

        encoder_outputs, encoder_attn_weights = self.encoder(src,src_mask)

        decoder_outputs, _ , enc_dec_mha_attn_weights = self.decoder(trg, encoder_outputs, trg_mask, src_mask)


        logits = self.linear_layer(decoder_outputs)
        # shape(logits) = [B x TRG_seq_len x TRG_vocab_size]

        return logits
'''

st.code(code, language='python')
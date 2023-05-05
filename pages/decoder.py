import streamlit as st
import numpy as np
import pandas as pd


# @st.cache

st.markdown("## Decoder")

import streamlit as st
import numpy as np
import pandas as pd


# @st.cache


st.write('''
#### Encoder-Decorder multihead attention in Decoder
This multihead attention is called encoder-decorder multihead attention. For this multihead attention we create we create key and value vectors from the encoder output. Query is created from the output of previous decoder layer.

enc_dec_mha , enc_dec_attn_weights = self.enc_dec_mha(norm1, encoder_outputs, encoder_outputs, mask=src_mask)
''')

code = '''


class Mha(nn.Module):
    def __init__(self, num_heads = 2, embed_dim = 4, dropout = 0.3):
        super().__init__()

        self.embed_dim = embed_dim 
        self.num_heads = num_heads

        self.single_head_dim = self.embed_dim // self.num_heads

        self.dropout = nn.Dropout(dropout)

        self.linear_Qs = nn.ModuleList([nn.Linear(embed_dim, self.single_head_dim)
        for _ in range(num_heads)])
        

        self.linear_Ks = nn.ModuleList([nn.Linear(embed_dim, self.single_head_dim)
        for _ in range(num_heads)])

        self.linear_Vs = nn.ModuleList([ nn.Linear(embed_dim, self.single_head_dim)
        for _ in range(num_heads)])

        self.mha_linear = nn.Linear(embed_dim, embed_dim)


    
    def scaled_dot_product_attention(self, Q, K, V, mask = None):
        ## shape Q , K, V: # of batch * [ seq * self.single_head_dim(embed_dim / num_heads))    ]
        ### q matmul k => batch * [ seq_len * seq_len ] (attention_wieght)
        ### output -- batch * [ seq * self.d]

        Q_K_matmul = torch.matmul(Q, K.permute(0,2,1))
        

        scores = Q_K_matmul / math.sqrt(self.single_head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask ==0 , -1)

        attention_weight = F.softmax(scores, dim = -1)

        output = torch.matmul(attention_weight, V)

        return output, attention_weight 

    
    def forward(self, q,k,v, mask= None):
        # shape x = batch * [ seq_len * embed_dim  ]

        Q = [linear_Q(q)   for linear_Q in self.linear_Qs]
        K = [linear_K(k)  for linear_K in self.linear_Ks]
        V = [linear_V(v)  for linear_V in self.linear_Vs]

        output_per_head = []
        attn_weights_per_head = []

        for Q_, K_, V_ in zip(Q,K,V):
            output, attn_weight = self.scaled_dot_product_attention(Q_, K_, V_)
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)

        output = torch.cat( 
            output_per_head, -1
        )
       
        attn_weights = torch.stack(attn_weights_per_head).permute(1,0,2,3)

        projection = self.dropout(self.mha_linear(output))

        return projection, attn_weight 



'''

st.code(code, language='python')

st.write('''

### Target mask for decoder


Mask is used because while creating attention of target words, we donot need a word to look in to the future words to check the dependency. ie, we already learned that why we create attention because we need to know contribution of each word with the other word. Since we are creating attention for words in target sequnce, we donot need a particular word to see the future words. For eg: in word "I am a strudent", we donot need the word "a" to look word "student".

''')

code = '''

class Decoder(nn.Module):
    def __init__(self, Embeddings, embed_dim, num_heads, num_layers, d_ff, device= 'cpu', dropout = 0.3):
        super().__init__()

        self.embedding = Embeddings

        self.PE = PositionalEncoding(embed_dim, device = device)

        self.dropout = nn.Dropout(dropout)

        self.decoders = nn.ModuleList( [DecoderLayer(
            embed_dim,
            num_heads,
            d_ff,
            dropout) for layer in range(num_layers)])

        
    def forward(self, x,  encoder_output,trg_mask, src_mask):
        embeddings = self.embedding(x)

        encoding = self.PE(embeddings)

        for decoder in self.decoders:
            encoding, masked_mha_attn_weights, enc_dec_mha_attn_weights = decoder(encoding, encoder_output, trg_mask, src_mask)
            # shape(encoding) = [B x TRG_seq_len x D]
            # shape(masked_mha_attn_weights) = [B x num_heads x TRG_seq_len x TRG_seq_len]
            # shape(enc_dec_mha_attn_weights) = [B x num_heads x TRG_seq_len x SRC_seq_len]

            return encoding, masked_mha_attn_weights, enc_dec_mha_attn_weights








'''


st.code(code, language='python')
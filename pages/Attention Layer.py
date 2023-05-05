import streamlit as st
import numpy as np
import pandas as pd


# @st.cache

st.markdown("## Attention Layer")
st.write('''
Attention layers are made up of n_heads heads. The heads act independently and additively, we just add their outputs together, and back to the stream.

In each head: There is a Key matrix,Query matrix and a Value matrix to generate key, query and value. These matrixes are learned during training.

After embedding and positional encoding our output will be of dimension (32,10,512) (batch_size, sequence_len, embed_dim)

Step 1: Trasformer paper set the n_heads = 8 We will resize (32,10,512) to (32, 10, 8, 64) 864=512 64 is the single head dimension. Each key,query, value matrix will be of 6464

Step 2: Calculate the score. ie, we will multiply query marix with key matrix. torch.matmul(Q, K.permute(0,2,1)) As Q,K have the same dimensions,we need to transpose K for multiplication convinience

Step 3: Now divide the output matrix with square root of dimension of key matrix and then apply Softmax over it.

Step 4: mutiply with value matrix

Compute that for every head, and then concatenate output
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
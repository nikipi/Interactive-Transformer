import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title


st.set_page_config(
    page_title="Transformer Overview",
    page_icon="👋",
)


 

import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 


 # import your pages here


add_page_title() 

show_pages(
    [
        Page("Transformer_Overview.py", "Transformer Overview", "🏠"),
        Page("pages/Embedding.py", "Embedding Layer", "1"),
        Page("pages/Positional Encoding Layer.py", "Positional Encoding Layer", "2"),
        Page("pages/Attention Layer.py", "Attention Layer", "3"),
        Page("pages/Feed Forward Layer.py", "Feed Forward Layer", "4"), 

        Page("pages/encoder.py", "Encoder", "5"),
        Page("pages/decoder.py", "Decoder", "6"),
        Page("pages/Transformer Code.py", "Transformer Code", ":books:"),
    ]
)

import graphviz

st.graphviz_chart('''
    digraph G {
    node [shape=record, style=filled, fillcolor=white];

    subgraph cluster_encoder {
        label="Encoder";
        URL="https://nikipi-interactive-transformer-transformer-overview-pli2pj.streamlit.app/Encoder"
        node [fillcolor=lightblue];
        input_embeddings -> mha[label="Positional Encoding"];
        mha -> ff1 [label="Add & Norm"];
             
        
    }
    subgraph cluster_decoder {
        label="Decoder";
        URL="https://nikipi-interactive-transformer-transformer-overview-pli2pj.streamlit.app/DEcoder"
        node [fillcolor=grey];
        dec1 [label="OutPut Embedding"];
        dec1 -> mmha [label="Positional Encoding"];
        mmha -> mha2 [label="Add & Norm"];
        ff1 -> mha2 [label="Add & Norm"];
        mha2 -> ff2 [label="Add & Norm"];
        ff2 -> lin [label="Add & Norm"];
        lin -> soft
       
        
    }
    input_embeddings [label="Input Embeddings"; URL="https://nikipi-interactive-transformer-transformer-overview-pli2pj.streamlit.app/Embedding%20Layer"];
    mha [label="Multi-Head Attention"];
    
    ff1 [label="Feed Forward"];
    mmha [label="Masked Multi-Head Attention"];
    mha2 [label="Multi-Head Attention"];
    ff2 [label="Feed Forward"];
    lin [label= "Linear"];
    soft [label= "Softmax"]
    


  
  
    
    rankdir=TR;
    
}

}

''')





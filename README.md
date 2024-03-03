<h2 align="center">LLaMA-02</h2>
<p align="center">pytorch implementation of H. Touvron  et al. (2023)</p>  
<br>

Llama 2[^1] resembles the encoder block from transformer architecture, which takes the input sentence as ordered tokens and predicts the next token.
<p align="center">
  <img src="/assets/pepper.png" alt="Transformer Network Architecture">
</p>

# Positional Embeddings

Positional embeddings are needed in transformers because they are invariant to the order by default[^2], A language is sequential in nature and order is essential to the semantics and syntax of a given sentence, without positional information the meaning of sentence is not well defined.

Positional information can be added by using positional embeddings, manipulating attention matrices or by processing the input with a recurrent neural network.

## Absolute Positional Embeddings

Absoulute positional embeddings encode the absoulute position of a word with a sentence. For a given sentence, positional information is represented in a vector of same length of the sentence. Each one of these vectors represent one specific position in a sentence. Vector representing the position information is added and their sum is fed to the Transformer model. In general positional embeddings are generated in two ways.

### Learned Positional Embeddings
The positional embeddings are treated as trainable parameters just like word embeddings. The model learns these positional embeddings during training alongside the other parameters.

### Fixed Positional Embeddings
 Positional embeddings is constructed for each possible position in the seqence, they are generated using predetermined mathematical functions, typically involving sine and cosine.

## Reative Positional Embeddings
Relative positional embeddings capture the distance and direction between pairs of words in a sequence. Here positional representation is learned for every pair of tokens in a sentence. 

## Rotary Positional Embeddings

Instead of directly adding a positional embedding vector to the word embedding, RoPE uses rotation matrices to encode the position.[^3] 



[^1]:H. Touvron _et al._, “Llama 2: Open Foundation and Fine-Tuned Chat Models.” arXiv, Jul. 19, 2023. Available: [http://arxiv.org/abs/2307.09288](http://arxiv.org/abs/2307.09288). [Accessed: Mar. 02, 2024]

[^2]:P. Dufter, M. Schmitt, and H. Schütze, “Position Information in Transformers: An Overview,” _Computational Linguistics_, vol. 48, no. 3, pp. 733–763, Sep. 2022, doi: [10.1162/coli_a_00445](https://doi.org/10.1162/coli_a_00445)

[^3]:J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding.” arXiv, Nov. 08, 2023. Available: [http://arxiv.org/abs/2104.09864](http://arxiv.org/abs/2104.09864). [Accessed: Mar. 03, 2024]
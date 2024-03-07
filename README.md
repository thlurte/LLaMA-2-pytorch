<h2 align="center">LLaMA-02</h2>
<p align="center">pytorch implementation of H. Touvron  et al. (2023)</p>  
<br>

Llama 2[^1] resembles the encoder block from transformer architecture, which takes the input sentence as ordered tokens and predicts the next token.
<p align="center">
  <img src="/assets/pepper.png" alt="Transformer Network Architecture">
</p>

## Positional Embeddings

Positional embeddings are needed in transformers because they are invariant to the order by default[^2], A language is sequential in nature and order is essential to the semantics and syntax of a given sentence, without positional information the meaning of sentence is not well defined.

Positional information can be added by using positional embeddings, manipulating attention matrices or by processing the input with a recurrent neural network.

### Absolute Positional Embeddings

Absoulute positional embeddings encode the absoulute position of a word with a sentence. For a given sentence, positional information is represented in a vector of same length of the sentence. Each one of these vectors represent one specific position in a sentence. Vector representing the position information is added and their sum is fed to the Transformer model. In general positional embeddings are generated in two ways.

#### Learned Positional Embeddings
The positional embeddings are treated as trainable parameters just like word embeddings. The model learns these positional embeddings during training alongside the other parameters.

#### Fixed Positional Embeddings
 Positional embeddings is constructed for each possible position in the seqence, they are generated using predetermined mathematical functions, typically involving sine and cosine.

### Reative Positional Embeddings
Relative positional embeddings capture the distance and direction between pairs of words in a sequence. Here positional representation is learned for every pair of tokens in a sentence. 

### Rotary Positional Embeddings
Instead of directly adding a positional embedding vector to the word embedding, RoPE uses rotation matrices to encode the position.[^3] 


Based on the paper:

$$
R^d_{\Theta,m}x=
\begin{pmatrix}x_1\\x_2\\x_3\\x_4\\.\\.\\.\\x_{d-1}\\x_d
\end{pmatrix} 
\bigotimes 
\begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \\ .\\.\\.\\\cos m\theta_{d/2}\\ \cos m\theta_{d/2}
\end{pmatrix}
+
\begin{pmatrix}-x_2\\x_1\\-x_4\\x_3\\.\\.\\.\\-x_d\\x_{d-1}\end{pmatrix}\bigotimes\begin{pmatrix}\sin m\theta_1\\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2\\ . \\ . \\ . \\\sin m\theta_{d/2} \\ \sin m\theta_{d/2} \end{pmatrix}
$$

where :
- $\Theta = \{\theta_i=10000^{-2(i-1)/d},i\varepsilon [1,2,...,d/2]\}$
- $m=[1,2,...,\text{seq\_len}]$

A slight different way to put this would be, We perform outer product on `m` and $\Theta$  and get a matrix that looks like this 

$$
\Theta \otimes \mathbf{m} = 
\begin{pmatrix} \theta_1 \cdot m_1 & \theta_1 \cdot m_2 & \cdots & \theta_1 \cdot m_{seq\_len} \\ 
\theta_2 \cdot m_1 & \theta_2 \cdot m_2 & \cdots & \theta_2 \cdot m_{seq\_len} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\theta_{d/2} \cdot m_1 & \theta_{d/2} \cdot m_2 & \cdots & \theta_{d/2} \cdot m_{seq\_len} 
\end{pmatrix}
$$

Then we convert this matrix into complex form

$$
\Theta \otimes \mathbf{m} = 
\begin{pmatrix} 
\cos( m_1 \theta_1 ) + i \sin( m_1 \theta_1 ) & \cos (m_2 \theta_1 ) + i \sin( m_2 \theta_1 ) & \cdots & \cos( m_{seq\_len} \theta_1 ) + i \sin( m_{seq\_len} \theta_1 ) \\ 
\cos( m_1 \theta_2 ) + i \sin( m_1 \theta_2 ) & \cos( m_2 \theta_2 ) + i \sin( m_2 \theta_2 ) & \cdots & \cos( m_{seq\_len} \theta_2 ) + i \sin( m_{seq\_len} \theta_2 ) \\
\vdots & \vdots & \ddots & \vdots \\
\cos( m_1 \theta_{d/2}) + i \sin( m_1 \theta_{d/2} ) & \cos( m_2 \theta_{d/2} ) + i \sin( m_2 \theta_{d/2} ) & \cdots & \cos( m_{seq\_len} \theta_{d/2} ) + i \sin( m_{seq\_len} \theta_{d/2} ) 
\end{pmatrix}
$$

#### Explanation

To illustrate how this operation works, we will consider a simple embedding vector.

$$
\begin{bmatrix}
x_1 \\ 
x_2 \\ 
x_3 \\ 
x_4
\end{bmatrix}
$$

We reshape this vector by grouping two successive tokens.

$$
\begin{bmatrix}
[x_1 & x_2] \\
[x_3 & x_4] 
\end{bmatrix}
$$

In this matrix, given that we consider the first element, We represent this element in complex form where $x_1$ will be real part and $x_2$ will be the imaginary part. :

$$
\begin{bmatrix}
x_1+ix_2 \\  x_3 + ix_4 \\
\end{bmatrix}
$$

Then we perform element wise multiplication on matrix we obtained from $\Theta \otimes m$ 

$$
\begin{bmatrix}
x_1+ix_2 \\ x_3 + ix_4
\end{bmatrix}
\odot
\begin{bmatrix}
\cos(m_1\theta_1)+i\sin(m_1\theta_1) \\
\cos(m_1\theta_2)+i\sin(m_1\theta_2) \\
\end{bmatrix}
$$

$$
\begin{bmatrix}
x_1 \cos( m_1 \theta_1 ) - x_2 \sin( m_1 \theta_1 ) + i ( x_1 \sin( m_1 \theta_1 )+x_2 \cos( m_1 \theta_1 )) \\
x_3 \cos( m_1 \theta_2 ) - x_4 \sin( m_1 \theta_2 ) + i ( x_3 \sin( m_1 \theta_2 )+ x_4 \cos( m_1 \theta_2 )) \\
\end{bmatrix}
$$

Once computed, now we can split the real and imaginary part and flatten them to get the vector embedding with positional information being added.

$$
\begin{bmatrix}
x_1 \cos( m_1 \theta_1 ) - x_2 \sin( m_1 \theta_1 ) & x_1 \sin( m_1 \theta_1 ) + x_2 \cos( m_1 \theta_1 ) \\
x_3 \cos( m_1 \theta_2 ) - x_4 \sin( m_1 \theta_2 ) & x_3 \sin( m_1 \theta_2 ) + x_4 \cos( m_1 \theta_2 ) \\
\end{bmatrix}
$$

$$
\begin{bmatrix}
x_1 \cos( m_1 \theta_1 ) - x_2 \sin( m_1 \theta_1 ) \\ 
x_1 \sin( m_1 \theta_1 ) + x_2 \cos( m_1 \theta_1 ) \\
x_3 \cos( m_1 \theta_2 ) - x_4 \sin( m_1 \theta_2 ) \\ 
x_3 \sin( m_1 \theta_2 ) + x_4 \cos( m_1 \theta_2 ) \\
\end{bmatrix}
$$


[^1]:H. Touvron _et al._, “Llama 2: Open Foundation and Fine-Tuned Chat Models.” arXiv, Jul. 19, 2023. Available: [http://arxiv.org/abs/2307.09288](http://arxiv.org/abs/2307.09288). [Accessed: Mar. 02, 2024]

[^2]:P. Dufter, M. Schmitt, and H. Schütze, “Position Information in Transformers: An Overview,” _Computational Linguistics_, vol. 48, no. 3, pp. 733–763, Sep. 2022, doi: [10.1162/coli_a_00445](https://doi.org/10.1162/coli_a_00445)

[^3]:J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding.” arXiv, Nov. 08, 2023. Available: [http://arxiv.org/abs/2104.09864](http://arxiv.org/abs/2104.09864). [Accessed: Mar. 03, 2024]
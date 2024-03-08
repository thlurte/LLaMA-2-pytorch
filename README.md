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

<p align="center">
  <img src="/assets/a.png" alt="Eq">
</p>


where :

$$
\Theta = \{\theta_i=10000^{-2(i-1)/d},i\varepsilon [1,2,...,d/2]\}
$$

$$
m=[1,2,...,seq\_len]
$$

A slight different way to put this would be, We perform outer product on `m` and $\Theta$  and get a matrix that looks like this 

$$
\begin{align}
\Theta \otimes \mathbf{m} = 
\begin{pmatrix} \theta_1 \cdot m_1 & \theta_1 \cdot m_2 & \cdots & \theta_1 \cdot m_{seq\_len} \\ 
\theta_2 \cdot m_1 & \theta_2 \cdot m_2 & \cdots & \theta_2 \cdot m_{seq\_len} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\theta_{d/2} \cdot m_1 & \theta_{d/2} \cdot m_2 & \cdots & \theta_{d/2} \cdot m_{seq\_len} 
\end{pmatrix}
\end{align}
$$

Then we convert this matrix into complex form

$$
\begin{align}
\Theta \otimes \mathbf{m} = 
\begin{pmatrix} 
\cos( m_1 \theta_1 ) + i \sin( m_1 \theta_1 ) & \cos (m_2 \theta_1 ) + i \sin( m_2 \theta_1 ) & \cdots & \cos( m_{seq\_len} \theta_1 ) + i \sin( m_{seq\_len} \theta_1 ) \\ 
\cos( m_1 \theta_2 ) + i \sin( m_1 \theta_2 ) & \cos( m_2 \theta_2 ) + i \sin( m_2 \theta_2 ) & \cdots & \cos( m_{seq\_len} \theta_2 ) + i \sin( m_{seq\_len} \theta_2 ) \\
\vdots & \vdots & \ddots & \vdots \\
\cos( m_1 \theta_{d/2}) + i \sin( m_1 \theta_{d/2} ) & \cos( m_2 \theta_{d/2} ) + i \sin( m_2 \theta_{d/2} ) & \cdots & \cos( m_{seq\_len} \theta_{d/2} ) + i \sin( m_{seq\_len} \theta_{d/2} ) 
\end{pmatrix}
\end{align}
$$

#### Explanation

To illustrate how this operation works, we will consider a simple embedding vector.

$$
\begin{align}
\begin{bmatrix}
x_1 \\ 
x_2 \\ 
x_3 \\ 
x_4
\end{bmatrix}
\end{align}
$$

We reshape this vector by grouping two successive tokens.

$$
\begin{align}
\begin{bmatrix}
[x_1 & x_2] \\
[x_3 & x_4] 
\end{bmatrix}
\end{align}
$$

In this matrix, given that we consider the first element, We represent this element in complex form where $x_1$ will be real part and $x_2$ will be the imaginary part. :

$$
\begin{align}
\begin{bmatrix}
x_1 + i x_2 \\ 
x_3 + i x_4 \\
\end{bmatrix}
\end{align}
$$

Then we perform element wise multiplication on matrix we obtained from $\Theta \otimes m$ 

$$
\begin{align}
\begin{bmatrix}
x_1+ix_2 \\ 
x_3 + ix_4
\end{bmatrix}
\odot
\begin{bmatrix}
\cos(m_1\theta_1)+i\sin(m_1\theta_1) \\
\cos(m_1\theta_2)+i\sin(m_1\theta_2) \\
\end{bmatrix}
\end{align}
$$

$$
\begin{align}
\begin{bmatrix}
x_1 \cos( m_1 \theta_1 ) - x_2 \sin( m_1 \theta_1 ) + i ( x_1 \sin( m_1 \theta_1 )+x_2 \cos( m_1 \theta_1 )) \\
x_3 \cos( m_1 \theta_2 ) - x_4 \sin( m_1 \theta_2 ) + i ( x_3 \sin( m_1 \theta_2 )+ x_4 \cos( m_1 \theta_2 )) \\
\end{bmatrix}
\end{align}
$$

Once computed, now we can split the real and imaginary part and flatten them to get the vector embedding with positional information being added.

$$
\begin{align}
\begin{bmatrix}
x_1 \cos( m_1 \theta_1 ) - x_2 \sin( m_1 \theta_1 ) & x_1 \sin( m_1 \theta_1 ) + x_2 \cos( m_1 \theta_1 ) \\
x_3 \cos( m_1 \theta_2 ) - x_4 \sin( m_1 \theta_2 ) & x_3 \sin( m_1 \theta_2 ) + x_4 \cos( m_1 \theta_2 ) \\
\end{bmatrix}
\end{align}
$$

$$
\begin{align}
\begin{bmatrix}
x_1 \cos( m_1 \theta_1 ) - x_2 \sin( m_1 \theta_1 ) \\ 
x_1 \sin( m_1 \theta_1 ) + x_2 \cos( m_1 \theta_1 ) \\
x_3 \cos( m_1 \theta_2 ) - x_4 \sin( m_1 \theta_2 ) \\ 
x_3 \sin( m_1 \theta_2 ) + x_4 \cos( m_1 \theta_2 ) \\
\end{bmatrix}
\end{align}
$$

## Normalization

In general, due to the nature of deep neural networks,  the distribution of inputs to each layer changes during training as the parameters of the previous layers are updated.[^4] This tends to slow down the training process by making it mandate for us to choose a much lower learning rate.

We tend to address this issue by attempting to stabilize the distribution of layer activations throughout training.

### Batch Normalization

Batch normalization is applied to individual layers. In each training iteration each batch of input is normalized by subtracting their mean ($\mu_{\beta}$) and dividing by their standard deviation $\sigma^2_{\beta}$, where both are estimated based on the statistics of the current batch, and then we apply a scale coefficient ( $\gamma$ ) and an offset ( $\beta$ ) to recover the lost degrees of freedom.[^5][^8]

$$
\mu_{\beta}=\frac{1}{m}\sum^m_{i=1}x^i
$$

$$
\sigma^2_{\beta}=\frac{1}{m}\sum^m_{i=1}(x_i-\mu_{\beta})^2
$$

$$
\hat x_i=\frac{x_i-\mu_{\beta}}{\sqrt{\sigma^2_{\beta}+\epsilon}}
$$

$$
y_i=\gamma \hat x_i
+\beta
$$

### Layer Normalization

Instead of computing mean using a distribution of minibtach like we do with batch normalizatiom, we compute mean by averaging over the individual vector within specific layer. [^6][^7]


$$
\mu^l=\frac{1}{H}\sum^H_{i=1}a^l_i
$$

$$
\sigma^l=\sqrt{\frac{1}{H}\sum^H_{i=1}(a^l_i-u^l)^2}
$$

$$
y^l = \gamma^l \cdot \frac{a^l - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta^l
$$

### Root Mean Square Layer Normalization

What RMS proposes is that re-scaling is alone sufficient to help model converge faster, thus resulting in a more simplified method.[^9]

$$
\bar a_i=\frac{a_i}{RMS(a)}g_i,
$$

where:

$$
RMS(a)=\sqrt{\frac{1}{n}\sum^n_{i=1}a^2_i}
$$





[^1]:H. Touvron _et al._, “Llama 2: Open Foundation and Fine-Tuned Chat Models.” arXiv, Jul. 19, 2023. Available: [http://arxiv.org/abs/2307.09288](http://arxiv.org/abs/2307.09288). [Accessed: Mar. 02, 2024]
[^2]:P. Dufter, M. Schmitt, and H. Schütze, “Position Information in Transformers: An Overview,” _Computational Linguistics_, vol. 48, no. 3, pp. 733–763, Sep. 2022, doi: [10.1162/coli_a_00445](https://doi.org/10.1162/coli_a_00445)
[^3]:J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding.” arXiv, Nov. 08, 2023. Available: [http://arxiv.org/abs/2104.09864](http://arxiv.org/abs/2104.09864). [Accessed: Mar. 03, 2024]
[^4]: S. Ioffe and C. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” arXiv, Mar. 02, 2015. doi: [10.48550/arXiv.1502.03167](https://doi.org/10.48550/arXiv.1502.03167). Available: [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167). [Accessed: Mar. 08, 2024]
[^5]:“8.5. Batch Normalization — Dive into Deep Learning 1.0.3 documentation.” Available: [https://d2l.ai/chapter_convolutional-modern/batch-norm.html#batch-normalization](https://d2l.ai/chapter_convolutional-modern/batch-norm.html#batch-normalization). [Accessed: Mar. 08, 2024]
[^6]:J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer Normalization,” _arXiv.org_, Jul. 21, 2016. Available: [https://arxiv.org/abs/1607.06450v1](https://arxiv.org/abs/1607.06450v1). [Accessed: Mar. 08, 2024]
[^7]:_Layer Normalization | Lecture 63 (Part 2) | Applied Deep Learning_, (May 07, 2021). Available: [https://www.youtube.com/watch?v=eyPZ9Mrhri4](https://www.youtube.com/watch?v=eyPZ9Mrhri4). [Accessed: Mar. 08, 2024]
[^8]:“Normalizing Activations in a Network (C2W3L04) - YouTube.” Available: [https://www.youtube.com/watch?v=tNIpEZLv_eg&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=28&t=159s](https://www.youtube.com/watch?v=tNIpEZLv_eg&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=28&t=159s). [Accessed: Mar. 08, 2024]
[^9]:B. Zhang and R. Sennrich, “Root Mean Square Layer Normalization,” _arXiv.org_, Oct. 16, 2019. Available: [https://arxiv.org/abs/1910.07467v1](https://arxiv.org/abs/1910.07467v1). [Accessed: Mar. 08, 2024]
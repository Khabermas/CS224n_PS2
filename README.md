# ﻿Neural Transition-Based Dependency Parser


#### **Introduction**

The second part -- and primary coding exercise -- of PS2 involves construction of a Neural Transition-Based Dependency Parser (Neural TBDP).  (The first section can be viewed as preparation for that task). Dependency parsing is described, for example, in [Chapter 14](https://web.stanford.edu/~jurafsky/slp/14.pdf) of the draft 3rd edition of _Speech and Language Processing_, by Jurafsky and Martin.  Deterministic Transition-Based Dependency Parsing is discussed in lecture 6 and in several references below.

**Background**

During the first decade of this century, quite a lot of success was achieved with feature-based discriminative dependency parsers.  Transition-based parsers proved the most efficient.  But those parsers required expert-designed feature sets, which engender millions of sparse indicator features (with inefficiently-learned weights).

The first efficient  Neural TBDP was presented by Chen and Manning (2014). The NN learns a small number of compact dense vector representations of words, part-of-speech (POS) tags, and dependency labels.

The setup is summarized below.

All embeddings 50-dimensional

Input layer: x<sup>w</sup>, x<sup>t</sup>, x<sup>l</sup> where
  
x<sup>w</sup>: 18 features; best results start with pre-trained word embeddings
x<sup>t</sup>: 18 features; POS tags
x<sup>l</sup>: 12 features; Arc (dependency) labels


Hidden layer h = ( W<sub>1</sub><sup>w</sup>x<sup>w</sup> + W<sub>1</sub><sup>t</sup>x<sup>t</sup> + W<sub>1</sub><sup>l</sup>x<sup>l</sup> )<sup>3</sup>

	baseline model employs 200 hidden units

p = softmax(W<sub>2</sub>h)

Dropout (50%) as well as L2 regularization (embedding matrices, weight matrices, bias term)













![Chen and Manning (2014) configuration](https://github.com/Khabermas/CS224n_PS2/tree/master/images/CMc.jpg)
Chen and Manning (2014) configuration


The unusual and computationally-demanding cubic activation function provided a 0.8 to 1.2% improvement in UAS over tanh or sigmoid non-linearities.  The authors speculate that the cubic function can model product terms of any two or three elements directly.

The authors found that use of pre-trained word embeddings improved UAS by ≈ 0.7%.
Including POS embedding features in training added 1.7%, while (arc) label embedding helped by only 0.3%.  “This may be because the POS tags of two tokens already capture most of the label information between them.”


**The Assignment 2 Network Configuration**

The configuration is similar to that of Chen & Manning.  Differences include the following:

There are 36 (50-dimensional) input features, rather than 48.  Word and POS embeddings are included, but the less-useful arc label features are omitted.

The activation function is ReLU.

The input word and (for training) POS feature vectors are concatenated – not treated as separate inputs.  Hence there is a single weight matrix in the hidden layer.

Initialization is done differently (we employ Xavier initialization); C&M used some pre-computation tricks to speed training, while our smaller feature set, less-demanding activation function, and some other simplifications rendered that less necessary.

In the baseline configuration we use 200 hidden units (as did C&M).  There is no explicit regularization term in the loss function.  The principal means to prevent over-fitting are dropout (50%) and using parameters that gave the lowest validation set error in for the final evaluation against the test set – a sort of early stopping.

But in subsequent trials, L2-regularization (of the hidden layer and softmax layer weight matrices) is explored.









![Our configuration](https://github.com/Khabermas/CS224n_PS2/tree/master/images/lecture6_472.jpg)

**Assignment 2 Parser Results and Experiments**

Training the baseline model takes about 30 minutes on an oldish computer ( i7-2670QM CPU@2.20GHz X 8 with 5.7 GB RAM).  The machine is incapable of meaningful GPU support.  However, as TensorFlow was installed from source, it can make use of some streaming SIMD extensions in the intel chips (SSE3, SSE4, AVX) that may boost performance.  It is gratifying to see all CPUs running near capacity, in contrast to the unparallelized python programs of Assignment 1.

C&M achieved a UAS of 92.0% with the optimal configuration.
Our baseline parser setup yields 88.79% against the test set.

The effect of L2-regularization was explored, keeping the number of hidden state units at 200.  Results are summarized in the Table below.  The optimal regularization constant (1x10-9) provides a very modest improvement – test UAS 88.89%.

Changing the number of hidden units proved more interesting (second Table).
With 100 hidden units test UAS falls by 1.1%.  Conversely, increasing the number of hidden units boosts results substantially – with the benefit plateauing at 600-800 hidden units. With 800 hidden units the unregularized test UAS is 89.83 (and virtually as good with 600); that represents a more than 1.0% improvement from the baseline.  Additional hidden units begin to impair performance.

Configurations with more hidden units require longer to train – 60 to 90 minutes for 600 to 1000 units.  Hence the parameter space has not been much explored.  However, using 800 hidden units and L2-regularization (1e-9) attains a test set UAS of 89.97%.

It plan to explore configurations with a second hidden layer.


**Effect of L2 Regularization**

| lambda      | Best dev UAS  |  Test UAS   |
| ----------- | :-----------: | :---------: |
| 0           |    88.41      |    88.79    |
|             |               |             |
| 5e-10       |    88.57      |    88.86    |
|             |               |             |
| 1e-9        |    88.41      |  **88.89**  |
|             |               |             |
| 3e-9        |    88.27      |    88.80    |
|             |               |             |
| 1e-8        |    87.79      |    88.20    |



**Effect of Number of Hidden Units**

| h size              | Best dev UAS |  Test UAS   |
| ------------------- | :----------: |:-----------:|
|  100 §              |   87.27      |    87.67    |
|                     |              |             |
|  200                |   88.41      |    88.79    |
|                     |              |             |
|  400                |   88.93      |    89.39    |
|                     |              |             |
|  600 §              |   89.64      |    89.82    |
|                     |              |             |
|  800                |   89.32      |    89.83    |
|                     |              |             |
|  800 (l2reg = 1e-9) |   89.42      |  **89.97**  |
|                     |              |             |
| 1000 (l2reg = 1e-9) |   89.41      |    89.61    |

 
 § required more than 10 training epochs to reach best dev set UAS.







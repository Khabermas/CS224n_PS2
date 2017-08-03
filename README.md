# ﻿Neural Transition-Based Dependency Parser


## **Introduction**

The second part -- and primary coding exercise -- of PS2 involves construction of a Neural Transition-Based Dependency Parser (Neural TBDP).  (The first section can be viewed as preparation for that task). Dependency parsing is described, for example, in [Chapter 14](https://web.stanford.edu/~jurafsky/slp/14.pdf) of the draft 3rd edition of _Speech and Language Processing_, by Jurafsky and Martin.  Deterministic Transition-Based Dependency Parsing is discussed in lecture 6 and in several references below.

## **Background**

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























![Chen and Manning (2014) configuration](https://github.com/Khabermas/CS224n_PS2/blob/master/images/CMc.jpg)

       Chen and Manning (2014) configuration


The unusual and computationally-demanding cubic activation function provided a 0.8 to 1.2% improvement in UAS over tanh or sigmoid non-linearities.  The authors speculate that the cubic function can model product terms of any two or three elements directly.

The authors found that use of pre-trained word embeddings improved UAS by ≈ 0.7%.
Including POS embedding features in training added 1.7%, while (arc) label embedding helped by only 0.3%.  “This may be because the POS tags of two tokens already capture most of the label information between them.”
  


## **The Assignment 2 Network Configuration**

The configuration is similar to that of Chen & Manning.  Differences include the following:

There are 36 (50-dimensional) input features, rather than 48.  Word and POS embeddings are included, but the less-useful arc label features are omitted.

The activation function is ReLU.

The input word and (for training) POS feature vectors are concatenated – not treated as separate inputs.  Hence there is a single weight matrix in the hidden layer.

Initialization is done differently (we employ Xavier initialization); C&M used some pre-computation tricks to speed training, while our smaller feature set, less-demanding activation function, and some other simplifications rendered that less necessary.

In the baseline configuration we use 200 hidden units (as did C&M).  There is no explicit regularization term in the loss function.  The principal means to prevent over-fitting are dropout (50%) and using parameters that gave the lowest validation set error in for the final evaluation against the test set – a sort of early stopping.

But in subsequent trials, L2-regularization (of the hidden layer and softmax layer weight matrices) is explored.


















![Our configuration](https://github.com/Khabermas/CS224n_PS2/blob/master/images/lecture6_472.jpg)
  
  

## **Assignment 2 Parser Results and Experiments**

Training the baseline model takes about 30 minutes on an oldish computer ( i7-2670QM CPU@2.20GHz X 8 with 5.7 GB RAM).  The machine is incapable of meaningful GPU support.  However, as TensorFlow was installed from source, it can make use of some streaming SIMD extensions in the intel chips (SSE3, SSE4, AVX) that may boost performance.  It is gratifying to see all CPUs running near capacity, in contrast to the unparallelized python programs of Assignment 1.

C&M achieved a UAS of 92.0% with the optimal configuration.
Our baseline parser setup yields 88.79% against the test set.

The effect of L2-regularization was explored, keeping the number of hidden state units at 200.  Results are summarized in Table 1 below.  The optimal regularization constant (1x10-9) provides a very modest improvement – test UAS 88.89%.

Changing the number of hidden units proved more interesting (Table 2).
With 100 hidden units test UAS falls by 1.1%.  Conversely, increasing the number of hidden units boosts results substantially – with the benefit plateauing at 600-800 hidden units. With 800 hidden units the unregularized test UAS is 89.83 (and virtually as good with 600); that represents a more than 1.0% improvement from the baseline.  Additional hidden units begin to impair performance.

Configurations with more hidden units require longer to train – 60 to 90 minutes for 600 to 1000 units.  Hence the parameter space has not been much explored.  However, using 800 hidden units and L2-regularization (1e-9) attains a test set UAS of 89.97%.


**Table 1 -- Effect of L2 Regularization**

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
  



**Table 2 -- Effect of Number of Hidden Units**

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
|  800 (l2reg = 2e-9) |   89.38      |    89.96    |
|                     |              |             |
|  800 †              |   89.23      |    89.74    |         |
|                     |              |             |
| 1000 (l2reg = 1e-9) |   89.41      |    89.61    |

 
 § required more than 10 training epochs to reach best dev set UAS.

 † l2reg = 1e-9; penalty on bias terms as well as weight matraces.

 


### **Activation Function**

Here we explore the use of different activation units.  With 200 hidden units, relu6 (ReLu with the maximum output limited to 6) may have provided slightly better results, but the effect was not consistent.  Other activation functions seemed inferior in this very small set of experiments.

Note: For all below, l2reg = 1e-9

| Activation Function |  h size  | Best dev UAS |  Test UAS   |
| ------------------- | -------- | :----------: | :---------: |
|  ReLu               |   200    |    88.41     |   88.89     |
|                     |          |              |             |
|  tanh               |   200    |    86.92     |   87.22     |
|                     |          |              |             |
|  elu                |   200    |    87.97     |   88.29     |
|                     |          |              |             |
|  relu6              |   200    |    88.59     | **89.09**   |
|                     |          |              |             |
|  relu6              |   800    |    89.42     |   89.91     |

  


### **Changing Dropout Probability**

These were run after the deeper models discussed in the next section, but seem best grouped with other one (hidden) layer models.
Recall that the *dropout* paramater indicates the probability of retaining connection to the next layer; dropout = 1.0 would be fully-connected.

Note: For all below, l2reg = 1e-9


|  h size  |  dropout  |  Best dev UAS |   Test UAS   |
| -------- | --------- | :-----------: | :----------: |
|   600    |   0.58    |     89.19     |     89.77    |
|          |           |               |              |
|   600    |   0.65    |     89.64     |     89.90    |
|          |           |               |              |
|   600    |   0.80    |     89.30     |     89.87    |
|          |           |               |              |
|   800    |   0.50    |     89.42     |   **89.97**  |  from above
|          |           |               |              |
|   800    |   0.60    |     89.33     |     89.88    |
|          |           |               |              |
|   800    |   0.65    |     89.58     |     89.86    |
|          |           |               |              |




### **Two or Three Hidden Layer Models**

To run with other program components, multilayer_q2_parser_model.py must be renamed q2_parser_model.py

Some notes:

  * Increasing the number of hidden layers did not improve performance. Too much time was expended exploring this parameter space.

  * For all of these configurations, the number of units in each hidden layer (of *that* model) is the same.

  * In contrast to many reports, dropout = 0.5 was inferior to higher retention probabilities.

  * Employing dropout between each hidden layer proved inferior to dropout only between the penultimate and final layer.

  * Typically the learning rate was set at 0.002 for two-layer models and 0.005 for three-layer models.

  * In the experiments reported below, l2-regularization was applied to the weight matrices (l2reg = 1e-9).
    But given the limited dropout and modest regularization constant, the primary guard against overfitting may be early stopping
    (testing on the parameters from the epoch that gave the highest dev set score).

Also the models with three hidden layers were investigated before those with two.  The former, and most of the latter, were run on an earlier -- more inelegant -- version of the program.  The revision should not make a significant difference to the test results (I believe).  In the table below, experiments run using the revised code are in italics.  For two cases, the trials with the older code were rerun on the new -- test UAS results were higher, but not markedly.


_**Two hidden layers**_

|  h size  |  dropout  | dropout applied |  Best dev UAS |   Test UAS   |
| -------- | --------- | --------------- | :-----------: | :----------: |
|          |           |                 |               |              |
|  400     |    0.5    |  only final     |     88.58     |    88.24     |
|          |           |                 |               |              |
|  400     |    0.8    |  both           |     87.79     |    87.72     |
|          |           |                 |               |              |
|  400     |    0.8    |  only final     |     88.58     |    88.85     |
| _400_    |   _0.8_   | _only final_    |    _88.81_    |   _89.34_    |
|          |           |                 |               |              |
| _600_    |   _0.65_  | _only final_    |    _88.57_    |   _88.97_    |
|          |           |                 |               |              |
|  600     |    0.8    |  only final     |     89.00     |    89.01     |
| _600_    |   _0.8_   | _only final_    |    _89.14_    |   _89.27_    |
|          |           |                 |               |              |
| _600_    |   _1.0_   | _no dropout_    |    _88.81_    |   _89.01_    |
|          |           |                 |               |              |
|  800     |    0.8    |  only final     |     88.84     |    88.96     |
|          |           |                 |               |              |


_**Three hidden layers**_

|  h size  |  dropout  | dropout applied |  Best dev UAS |   Test UAS   |
| -------- | --------- | --------------- | :-----------: | :----------: |
|          |           |                 |               |              |
|  300     |    0.5    |  every layer    |     84.30     |    84.86     |
|          |           |                 |               |              |
|  300     |    0.5    |  only final     |     88.26     |    88.62     |
|          |           |                 |               |              |
|  300     |    0.8    |  every layer    |     87.69     |    87.84     |
|          |           |                 |               |              |
|  300     |    0.8    |  2nd & 3rd      |     88.78     |    88.63     |
|          |           |                 |               |              |
|  300     |    0.8    |  only final     |     88.44     |    89.05     |
|          |           |                 |               |              |
|  400     |    0.5    |  only final     |     88.97     |    88.94     |
|          |           |                 |               |              |
|  400     |    0.8    |  only final     |     88.85     |    89.02     |
|          |           |                 |               |              |

  

## **References and Further Reading**

**Transition-Based Parsing**
Joakim Nivre
[transition.pdf](http://stp.lingfil.uu.se/~nivre/master/transition.pdf)



**Transition-based Dependency Parsing**
Algorithms for NLP Course.
7-11
Miguel Ballesteros
[TbparsingSmallCorrection.pdf](http://demo.clab.cs.cmu.edu/fa2015-11711/images/b/b1/TbparsingSmallCorrection.pdf)



**A Fast and Accurate Dependency Parser using Neural Networks**
Danqi Chen, Christopher D. Manning 
[emnlp2014](http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf)



**Structured Training for Neural Network Transition-Based Parsing**
David Weiss, Chris Alberti,  Michael Collins,  Slav Petrov (Google)
<https://pdfs.semanticscholar.org/90a8/a18e127b063468d31ac7e4d5dc9b68a13ac3.pdf>



**Globally Normalized Transition-Based Neural Networks**
Daniel Andor, Chris Alberti, David Weiss, Aliaksei Severyn, Alessandro Presta, Kuzman Ganchev, Slav Petrov and Michael Collins
Google Inc
<http://arxiv.org/abs/1603.06042v2>
<https://pdfs.semanticscholar.org/4be0/dd53aa1c751219fa6f19fed8a6324f6d2766.pdf>


**Transition-based Dependency Parsing Using Two Heterogeneous Gated Recursive Neural Networks**
Xinchi Chen, Yaqian Zhou, Chenxi Zhu, Xipeng Qiu, Xuanjing Huang
Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1879–1889,
[EMNLP215](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP215.pdf)


**Deep Biaffine Attention For Neural Dependency Parsing**
Timothy Dozat & Christopher D. Manning
[Under review as a conference paper at ICLR 2017](https://web.stanford.edu/~tdozat/files/iclr2016-deep-biaffine.pdf)


**Transition-Based Dependency Parsing with Stack Long Short-Term Memory**
Chris Dyer, Miguel Ballesteros, Wang Ling, Austin Matthews, Noah A. Smith
<http://anthology.aclweb.org/P/P15/P15-1033.pdf>



**Greedy Transition-based Dependency Parsing with Stack LSTMs**
Miguel Ballesteros _et al._
Computational Linguistics  doi: 10.1162/COLI_a_00285
Submission received: 16 September, 2015; revised version received: 6 April, 2016; accepted for publication: 14 June, 2016.
<http://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00285>



**Incremental Recurrent Neural Network Dependency Parser with Search-based Discriminative Training**
Majid Yazdani & James Henderson
Proceedings of the 19th Conference on Computational Language Learning, pages 142–152, 
Beijing, China, July 30-31, 2015.
<http://majid.yazdani.me/papers/CONLL15.pdf>




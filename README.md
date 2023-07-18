# I2CNet
Code for methods in the paper: Intra- and Inter-Channel Deep Convolutional Neural Network with Dynamic Label Smoothing for Multichannel Biosignal Analysis
## Architecture of I2CNet with dynamic label smoothing
![overall structure](fig/fig1.png)
>Detailed architecture of our proposed neural network. The proposed architecture mainly consists of a `feature extractor`, a `label predictor`, and `label adjustor` which not included in a standard feed-forward neural network. During training phase, both label predictor and label adjustor can supervise the feature extractor.
## Different components of I2CNET
### I2C Convolutional Block
<p align="left">
  <img src="fig/fig2.png" alt="模块1" width="45%">
  <br>
  在这里，你可以详细地介绍模块1的功能和目标。这个模块的输入和输出是什么？如果有的话，可以给出详细的数学或代码实现。
</p>


### I2C MSE Module
### I2C Attention Module

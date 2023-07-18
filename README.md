# I2CNet
Code for the machine learning methods and LFCNN in the paper: Intra- and Inter-Channel Deep Convolutional Neural Network with Dynamic Label Smoothing for Multichannel Biosignal Analysis
# Architecture of I2CNet with dynamic label smoothing
![overall structure](fig/fig1.png)
>Detailed architecture of our proposed neural network. The proposed architecture mainly consists of a feature extractor, a label predictor, and label adjustor (gray) which not included in a standard feed-forward neural network. During training phase, both label predictor and label adjustor can supervise the feature extractor.

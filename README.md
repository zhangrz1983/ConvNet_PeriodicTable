# ConvNet_PeriodicTable
Machine learning material properties from the periodic table using convolutional neural networks, Chemical Science, 2018, 9, 8426-8432, https://doi.org/10.1039/C8SC02648C

The original paper used Caffe, here reproduced using Keras and PyTorch. In the following work we will mainly use PyTorch. 

There some minor difference between the orginal paper and current implemenation 
1. In the original paper, smooth L1 loss was used, i.e. the Huber loss, which combines the advantages of L1 and L2 loss. Here MSE is used. 
2. In the orignial paper, the train/test loss convergence at around 20000 iterations/epoch. Here in the 'Results' folder, the PyTorch code only runs 1000 epoch to save time. The output accuracy at 1000 epoch is comparable to the results at 1000 epoch in the original paper. (But not as accurate as that at 20000 epoch)
3. The Periodic Table maxtirx is inintialized at -0.01, not -20 as in the original paper. 


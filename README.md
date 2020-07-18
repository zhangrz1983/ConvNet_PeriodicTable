# ConvNet_PeriodicTable
Machine learning material properties from the periodic table using convolutional neural networks, Chemical Science, 2018, 9, 8426-8432, https://doi.org/10.1039/C8SC02648C

comparable to ChemSci paper at 1000 epoch

1. Loss function should be changed to smooth l1 loss, i.e. the Huber loss, which combines the advantages of L1 and L2 loss
2. the train/test loss convergence at around 20000 iterations
3. should output test, add sys.stdout.flush(), 

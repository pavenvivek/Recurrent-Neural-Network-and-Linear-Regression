# Time series prediction using Recurrent Neural Network

run file as 
```
python3.7 recurrent_neural_networks.py
```

output_RNN.txt contains full output.

The program recurrent_neural_networks.py implements a Recurrent Neural Network from scratch (i.e. without using libraries such as TensorFlow or PyTorch). It trains the network using gradient descent with backpropagation. We generate a full cycle (period) of a sine wave that is 100 samples long and use it to train the model. During training, we feed thesignal into the network in minibatches of desired size. The trained network is then used to predict the next value in the time-series.

# Linear Regression

We have two data sets "pred1.dat" and "pred2.dat". The first has an n × d = 1000 × 50 data matrix (X) “pred1.dat” with a 1000 × 1 response vector (y) in “resp1.dat.” The second has a 1000 × 500 data matrix “pred2.dat” with a response vector in “resp2.dat.” These data sets were generated according to the standard linear regression model y = Xw + e, where X is an n × d matrix of predictor variables, y is an n−dimensional vector of response variables, and e ∼ N(0, σ^2 * I) (Gaussian distribution) is an n−dimensional vector of model errors.

For each data set, we use the first half of the data (observations i = 1, ..., n/2, all d predictors) to estimate values for w, ŵ. Next, for each data set, we use the estimate of w on the 2nd half of the data set (n/2 + 1, ..., n), to get estimated response variables, ŷ.

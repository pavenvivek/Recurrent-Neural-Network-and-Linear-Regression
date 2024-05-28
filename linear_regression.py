#!/usr/local/bin/python3
#
# linear_regression.py
#
# Submitted by : Paventhan Vivekanandan, username : pvivekan
#
#

#!/usr/local/bin/python3

import numpy as np

def read_data (filename):
    fh = open(filename, 'r')
    data = []

    for line in fh:
        values = line.split(" ")
        
        row = []
        for element in values:        
            row.append(float(element))
        
        data.append(row)

    matrix_m = np.matrix(data)

    return matrix_m


def calculate_weights(pred_X, resp_Y):
    
    # we calculate W = (X^T * X)^-1 * X^T * Y
    
    step1 = pred_X.T * pred_X  # step1 -> X^T * X
    step2 = step1.I * pred_X.T # step2 -> (X^T * X)^-1 * X^T
    W = step2 * resp_Y         # W -> (X^T * X)^-1 * X^T * Y
    
    return W


def calculate_error(pred_X, resp_Y, weights_W):

    # we calculate sum_{i=n/2+1}^{n} (resp_y - output_y)^2
    
    output_Y = pred_X * weights_W
    error = resp_Y - output_Y
    
    squared_error_sum = 0
    
    for i in range(0, error.size):
        squared_error_sum = squared_error_sum + \
            (error.item(i) * error.item(i))

    return squared_error_sum


def calculate_weights_and_error(pred_X, resp_Y):

    pred_X_fst_half = pred_X[0:int(pred_X.shape[0]/2)]
    resp_Y_fst_half = resp_Y[0:int(resp_Y.shape[0]/2)]
    
    W = calculate_weights(pred_X_fst_half, resp_Y_fst_half)
    print("weights -> {}".format(W))

    pred_X_sec_half = pred_X[int(pred_X.shape[0]/2):]
    resp_Y_sec_half = resp_Y[int(resp_Y.shape[0]/2):]

    E = calculate_error(pred_X_sec_half, resp_Y_sec_half, W)
    print("squared error -> {}".format(E))
    

if __name__ == "__main__":
    
    print("Inside Main !")
    
    pred1_X = read_data("pred1.dat")
    pred2_X = read_data("pred2.dat")
    resp1_Y = read_data("resp1.dat")
    resp2_Y = read_data("resp2.dat")

    calculate_weights_and_error(pred1_X, resp1_Y)
    calculate_weights_and_error(pred2_X, resp2_Y)

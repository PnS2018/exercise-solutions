"""Answer to Exercise 1.3

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function

import numpy as np
import keras.backend as K

# create variable w, b and x
w = K.placeholder(shape=(2,), dtype=np.float32)
# note that b is not a scalar
b = K.placeholder(shape=(1,), dtype=np.float32)
x = K.placeholder(shape=(2,), dtype=np.float32)

# compute the sum
a = K.sum(w*x+b)
# compute the function f
y = 1./(1+K.exp(-a))

# compile the function
fun = K.function(inputs=[x, w, b], outputs=[y])

# Test function
# compute y = f(2*x_1+3*x_2+0.5)

W = np.array([2, 3], dtype=np.float32)
B = np.array([0.5], dtype=np.float32)
X = np.array([1, 2], dtype=np.float32)

print (fun([X, W, B])[0])

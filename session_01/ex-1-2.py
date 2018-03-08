"""Answer to Exercise 1.2

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function

import numpy as np
import keras.backend as K

# get a scalar input x
x = K.placeholder(shape=(), dtype=np.float64)

# formulate tanh function
y = (K.exp(x)-K.exp(-x))/(K.exp(x)+K.exp(-x))

# compile function
tanh_fun = K.function(inputs=[x], outputs=[y])

# Test tanh function first
test_values = [-100, 1, 0, 1, 100]
for test_val in test_values:
    print ("tanh(", test_val, ")=", tanh_fun([test_val])[0])

# Compute the gradient of the tanh function respect to x
dydx = K.gradients(loss=y, variables=[x])

grad_tanh_fun = K.function(inputs=[x], outputs=[dydx[0]])

for test_val in test_values:
    print ("grad_tanh(", test_val, ")=", grad_tanh_fun([test_val])[0])

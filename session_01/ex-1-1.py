"""Answer to Exercise 1.1

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
# You need to import packages before you can use them
from __future__ import print_function  # this for Python 3 compatibility

import numpy as np  # import numpy for array support
import keras.backend as K  # import Keras backend

# create placeholders
a = K.placeholder(shape=(5,), dtype=np.float64)
b = K.placeholder(shape=(5,), dtype=np.float64)
c = K.placeholder(shape=(5,), dtype=np.float64)

# compute a^2+b^2+c^2+2bc
y = a**2+b**2+c**2+2*b*c

# compile the function
fun = K.function(inputs=[a, b, c], outputs=[y])

# test example
A = np.array([1, 2, 3, 4, 5], dtype=np.float64)
B = np.array([5, 4, 3, 2, 1], dtype=np.float64)
C = np.array([1, 1, 1, 1, 1], dtype=np.float64)

# Get output, the output is a list that has one element,
# extract that element by indexing [0]
output = fun([A, B, C])[0]

# print output
print (output)

# Run this script by
# $ python ex-1-1.py

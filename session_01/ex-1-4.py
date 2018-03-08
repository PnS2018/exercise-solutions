"""Answer to Exercise 1.4

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function

import numpy as np
import keras.backend as K

# define list of placeholders for variables
N = 3
theta = [K.placeholder(shape=(), dtype=np.float32) for i in range(N+1)]
x = K.placeholder(shape=(), dtype=np.float32)

# Compute function

y = theta[-1]
for i in range(N):
    y += theta[i]*x**(i+1)

# compile function
fun = K.function(inputs=theta+[x], outputs=[y])

# setup example
# y = theta_2*x^3+theta_1*x^2+theta_0*x+theta_3

Theta = [1, 2, 3, 4, 5]
X = 5

print (fun(Theta+[X])[0])

# Compute individual gradient

grad_collector = [K.gradients(y, th)[0] for th in theta]

grad_fun = K.function(inputs=theta+[x], outputs=grad_collector)

# Evaluate each gradient
print (grad_fun(Theta+[X]))

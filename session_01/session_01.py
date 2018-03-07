import keras.backend as K
# the backend can also be imported as
# from keras import backend as K
import numpy as np

# Exercise 1

# Required to create three symbolic placeholder tensors

a = K.placeholder(shape=(5,))
b = K.placeholder(shape=(5,))
c = K.placeholder(shape=(5,))

"""For the interested, a small hack using list comprehension is
a, b, c = [K.placeholder(shape=(5,)) for _ in range(3)]
The _ here represents a variable which is not used apart from looping"""

required_tensor = a ** 2 + b ** 2 + c ** 2 + 2 * b * c

required_function = K.function(inputs=(a, b, c), outputs=(required_tensor,))

print("Exercise 1 complete")

# Exercise 2

"""x can be initialized in various ways
It can be initialized as either
x = K.variable(value=np.array(1))
Or it can initialized through one of K.ones(shape=()) or K.zeros(shape=())"""
x = K.ones(shape=())

tanh_tensor = (K.exp(x) - K.exp(-x)) / (K.exp(x) + K.exp(-x))
gradient_tensor_list = K.gradients(loss=tanh_tensor, variables=(tanh_tensor,))
"""Note that the above variable is a list of tensors each which holds
the gradient of the tanh_tensor with respect to each of the variables given
as the input. In this case, since there is only one variable, the
gradient_tensor_list is a list with only element in it"""

required_function = K.function(inputs=(x,),
                               outputs=[tanh_tensor] + gradient_tensor_list)
"""Note that the input for the parameter outputs above can either be a list or
a tuple. In the use above, we added two lists which would be concatenation
operation in python which results in a single list holding the elements in both
the lists"""

print("Exercise 2 complete")

# Exercise 3

"""Required two variables and one placeholders
Note that the gradients can only be computed with respect to variables and
not placeholders"""
w, b = K.ones(shape=(2,)), K.ones(shape=(1,))
x = K.placeholder(shape=(1,))

"""You can define the given operations as
required_tensor = 1 / (1 + K.exp(-(w[0] * x[0] + w[1] * x[1] + b)))
like many of you have done. But there is an alternate way as well, especially
useful when the length of the tensors is more than 2"""

required_tensor = 1 / (1 + K.exp(-(K.sum(w * x) + b)))
required_gradient_list = K.gradients(loss=required_tensor, variables=(w,))

required_function = K.function(inputs=(w, b, x),
                               outputs=required_gradient_list)

"""Required to analyze the function at different values of x.
But before we analyze we must choose a set of values for w and b, let us choose
the values of [1, 1] and [1] for them."""

w_value = np.array([1., 1.])
b_value = np.array([1.])

for x_value in np.array([[-100], [-1], [0], [1], [100]]):
    print("The outputs of the function " \
          "at value {} is {}.".format(x_value,
                                      required_function((w_value,
                                                         b_value,
                                                         x_value))))

print("Exercise 3 complete")

# Exercise 4

# Required to create an n-degree polynomial for an arbitrary n
# There are multiple ways of doing this, one of the ways is written below

n = 5  # choosing some value

variable = K.ones(shape=(n + 1,))
x = K.placeholder(shape=())

output = variable[0]

for idx in range(1, n+1):
    output += variable[idx] * (x ** idx)

gradient_tensor_list = K.gradients(loss=output, variables=(variable,))

# Alternatively you can create a function to do this, with a parameter n

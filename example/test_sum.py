import numpy as np

a1 = np.array([[.1, .2, .3]])
a2 = np.array([[.5, .5, .5], [.1, .2, .3] ])
a3 = np.array([[.1, .2, .3], [.3, .4, .7], [1, 1, 1] ])
a4 = np.array([[.2, .3, .6], [.3, .4, .7], [.5, .5, .5], [1, 1, 1] ])

# Now we test it with some predictions
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    return e_x / sum

# print(a1.shape, a2.shape, a3.shape, a4.shape)

print( "Original", a1,  "softmax", softmax(a1), sep="\n" )

# print( a2, softmax(a2), softmax_ralf(a2), sep="\n" )

print( "Original", a3,  "softmax", softmax(a3), sep="\n" )

# print( a4, softmax(a4), softmax_ralf(a1),  sep="\n" )


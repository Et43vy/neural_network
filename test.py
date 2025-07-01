import numpy as np

n = [2, 3, 3, 1]
print("layer 0 / input layer size", n[0])
print("layer 1 size", n[1])
print("layer 2 size", n[2])
print("layer 3 size", n[3])

W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)
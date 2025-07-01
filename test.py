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

print("Weights for layer 1 shape:", W1.shape)
print("Weights for layer 2 shape:", W2.shape)
print("Weights for layer 3 shape:", W3.shape)
print("bias for layer 1 shape:", b1.shape)
print("bias for layer 2 shape:", b2.shape)
print("bias for layer 3 shape:", b3.shape)

#print(W1)
X = np.array([
    [150, 70], # it's our boy Jimmy again! 150 pounds, 70 inches tall. 
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])

print(X.shape) # prints (10, 2)
A0 = X.T
print(A0.shape)

y = np.array([
    0,  # whew, thank God Jimmy isn't at risk for cardiovascular disease.
    1,   # damn, this guy wasn't as lucky
    1, # ok, this guy should have seen it coming. 5"8, 312 lbs isn't great.
    0,
    0,
    1,
    1,
    0,
    1,
    0
])
m = 10

# we need to reshape to a n^[3] x m matrix
Y = y.reshape(n[3], m)
Y.shape

def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))

#print(sigmoid(np.array([-88,200,500,569,.2])))

m=10

# layer 1 calculations

Z1 = W1 @ A0 + b1
assert Z1.shape ==(n[1],m)
A1 = sigmoid(Z1)
print(A1)

#layer 2 calculations
Z2 = W2 @ A1 + b2
assert Z2.shape == (n[2],m)
A2 = sigmoid(Z2)
print(A2)

# layer 3 calculations
Z3 = W3 @ A2 + b3
assert Z3.shape == (n[3],m)
A3 = sigmoid(Z3)
print(A3)

print(A3.shape)
y_hat = A3
print(y_hat)
import numpy as np

#create network architecture
L = 3
n = [2, 3, 3, 1]
#print("layer 0 / input layer size", n[0])
#print("layer 1 size", n[1])
#print("layer 2 size", n[2])
#print("layer 3 size", n[3])

#create weights and biases
W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

#print("Weights for layer 1 shape:", W1.shape)
#print("Weights for layer 2 shape:", W2.shape)
#print("Weights for layer 3 shape:", W3.shape)
#print("bias for layer 1 shape:", b1.shape)
#print("bias for layer 2 shape:", b2.shape)
#print("bias for layer 3 shape:", b3.shape)

#print(W1)

# create training data and lables
def prepare_data():
    X = np.array([
        [150, 70],  
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
    y = np.array([0,1,1,0,0,1,1,0,1,0])
    m = 10
    A0 = X.T
    #print(A0.shape)
    Y = y.reshape(n[L],m)

    return A0, Y, m

# print(X.shape)

# we need to reshape to a n^[3] x m matrix
#Y = y.reshape(n[3], m)
#Y.shape
#m=10

#cost function
def cost(y_hat, y):
    losses = - (y * np.log(y_hat) + (1-y)*np.log(1 - y_hat))
    m = y_hat.reshape(-1).shape[0]
    summed_losses = (1/m) * np.sum(losses, axis = 1)
    return np.sum(summed_losses)

# create activation function
# def sigmoid(arr):
#    return 1 / (1 + np.exp(-1 * arr))

def g(z):
    return 1 / (1 + np.exp(-1*z))

#print(sigmoid(np.array([-88,200,500,569,.2])))

# create feed forward process
def feed_forward(A0):
    # layer 1 calculations
    Z1 = W1 @ A0 + b1
    #assert Z1.shape ==(n[1],m)
    A1 = g(Z1)
    #print(A1)

    #layer 2 calculations
    Z2 = W2 @ A1 + b2
    #assert Z2.shape == (n[2],m)
    A2 = g(Z2)
    #print(A2)

    # layer 3 calculations
    Z3 = W3 @ A2 + b3
    #assert Z3.shape == (n[3],m)
    A3 = g(Z3)
    #print(A3)

    cache = {
        "A0": A0,
        "A1": A1,
        "A2": A2
    }
    #print(A3.shape)
    y_hat = A3
    #print(y_hat)
    return y_hat, cache

A0, Y, m = prepare_data()

def backprop_layer_3(y_hat, Y, m, A2, W3):
  A3 = y_hat
  
  # step 1. calculate dC/dZ3 using shorthand we derived earlier
  dC_dZ3 = (1/m) * (A3 - Y)
  assert dC_dZ3.shape == (n[3], m)


  # step 2. calculate dC/dW3 = dC/dZ3 * dZ3/dW3 
  #   we matrix multiply dC/dZ3 with (dZ3/dW3)^T
  dZ3_dW3 = A2
  assert dZ3_dW3.shape == (n[2], m)

  dC_dW3 = dC_dZ3 @ dZ3_dW3.T
  assert dC_dW3.shape == (n[3], n[2])

  # step 3. calculate dC/db3 = np.sum(dC/dZ3, axis=1, keepdims=True)
  dC_db3 = np.sum(dC_dZ3, axis=1, keepdims=True)
  assert dC_db3.shape == (n[3], 1)

  # step 4. calculate propagator dC/dA2 = dC/dZ3 * dZ3/dA2
  dZ3_dA2 = W3 
  dC_dA2 = W3.T @ dC_dZ3
  assert dC_dA2.shape == (n[2], m)

  return dC_dW3, dC_db3, dC_dA2

def backprop_layer_2(propagator_dC_dA2, A1, A2, W2):
    # step 1. calculate dC/dZ2 = dC/dA2 * dA2/dZ2

  # use sigmoid derivation to arrive at this answer:
  #   sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
  #     and if a = sigmoid(z), then sigmoid'(z) = a * (1 - a)
  dA2_dZ2 = A2 * (1 - A2)
  dC_dZ2 = propagator_dC_dA2 * dA2_dZ2
  assert dC_dZ2.shape == (n[2], m)


  # step 2. calculate dC/dW2 = dC/dZ2 * dZ2/dW2 
  dZ2_dW2 = A1
  assert dZ2_dW2.shape == (n[1], m)

  dC_dW2 = dC_dZ2 @ dZ2_dW2.T
  assert dC_dW2.shape == (n[2], n[1])

  # step 3. calculate dC/db2 = np.sum(dC/dZ2, axis=1, keepdims=True)
  dC_db2 = np.sum(dC_dW2, axis=1, keepdims=True)
  assert dC_db2.shape == (n[2], 1)

  # step 4. calculate propagator dC/dA1 = dC/dZ2 * dZ2/dA1
  dZ2_dA1 = W2
  dC_dA1 = W2.T @ dC_dZ2
  assert dC_dA1.shape == (n[2], m)

  return dC_dW2, dC_db2, dC_dA1

    
def backprop_layer_1(propagator_dC_dA1, A1, A0, W1):

  # step 1. calculate dC/dZ1 = dC/dA1 * dA1/dZ1

  # use sigmoid derivation to arrive at this answer:
  #   sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
  #     and if a = sigmoid(z), then sigmoid'(z) = a * (1 - a)
  dA1_dZ1 = A1 * (1 - A1)
  dC_dZ1 = propagator_dC_dA1 * dA1_dZ1
  assert dC_dZ1.shape == (n[1], m)


  # step 2. calculate dC/dW1 = dC/dZ1 * dZ1/dW1 
  dZ1_dW1 = A0
  assert dZ1_dW1.shape == (n[0], m)

  dC_dW1 = dC_dZ1 @ dZ1_dW1.T
  assert dC_dW1.shape == (n[1], n[0])

  # step 3. calculate dC/db1 = np.sum(dC/dZ1, axis=1, keepdims=True)
  dC_db1 = np.sum(dC_dW1, axis=1, keepdims=True)
  assert dC_db1.shape == (n[1], 1)

  return dC_dW1, dC_db1

y_hat, cache = feed_forward(A0)
print(y_hat)

dC_dW3, dC_db3, dC_dA2 = backprop_layer_3(
  y_hat, 
  Y, 
  m, 
  A2 = cache["A2"], 
  W3 = W3
)

dC_dW3, dC_db3, dC_dA2 = backprop_layer_3(
   y_hat,
   Y,
   m,
   A2 = cache["A2"],
   W3 = W3
)

dC_dW2, dC_db2, dC_dA1 = backprop_layer_2(
    propagator_dC_dA2=dC_dA2, 
    A1=cache["A1"],
    A2=cache["A2"],
    W2=W2
)

dC_dW1, dC_db1 = backprop_layer_1(
    propagator_dC_dA1=dC_dA1, 
    A1=cache["A1"],
    A0=cache["A0"],
    W1=W1
)

def train():
   global W3, W2, W1, b3, b2 ,b1
   epochs = 1000000
   alpha = 0.001
   costs = []

   for e in range(epochs):
      # feed forward
      y_hat, cache = feed_forward(A0)

      # cost calculation
      error = cost(y_hat, Y)
      costs.append(error)

      # backprop
      dC_dW3, dC_db3, dC_dA2 = backprop_layer_3(
         y_hat,
         Y,
         m,
         A2 = cache["A2"],
         W3 = W3
      )

      dC_dW2, dC_db2, dC_dA1 = backprop_layer_2(
        propagator_dC_dA2=dC_dA2, 
        A1 = cache["A1"],
        A2 = cache["A2"],
        W2 = W2
      )

      dC_dW1, dC_db1 = backprop_layer_1(
        propagator_dC_dA1=dC_dA1, 
        A1=cache["A1"],
        A0=cache["A0"],
        W1=W1
     )
      
      #update weights
      W3 = W3 - (alpha * dC_dW3)
      W2 = W2 - (alpha * dC_dW2)
      W1 = W1 - (alpha * dC_dW1)

      b3 = b3 - (alpha * dC_db3)
      b2 = b2 - (alpha * dC_db2)
      b1 = b1 - (alpha * dC_db1)

      if e % 1000000 == 0:
         print(f"epoch {e}: cost = {error:4f}")
   return costs

costs = train()

y_hat, cache = feed_forward(A0)
print(y_hat)
print(Y)
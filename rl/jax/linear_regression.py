# %%
## Import

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random, nn
import matplotlib.pyplot as plt
from functools import reduce 
from timeit import timeit

# %%
# Generate test data for a linear equation
# y = mx + c

def generate(length = 10,
             coeffs = (2, 5),
             dtype=jnp.float32,
             key = random.PRNGKey(0)):
    x = random.normal(key, (length,), dtype=dtype)
    y = reduce(lambda acc, coeff: x * acc + coeff, coeffs, 0)
    noise = random.uniform(key, (length,), minval = 0, maxval = 1)
    return x, y + noise


# %%
# Data Generation

key = random.PRNGKey(0)
x, y = generate(length = 1000, coeffs = (2, 5, 3))
x1 = jnp.c_[jnp.ones(x.shape[0]), x, x * x]

loss_fn = jit(lambda w, x, y : jnp.mean((x @ w - y) ** 2))
update = jit(lambda w, x, y, lr=1e-2 : w - lr * grad(loss_fn)(w, x, y))

# %%
# Normal equation

loss_neq = [] 

def normal_eq(x, y):
    """
    Solve using normal equation

    Args:
      x (numpy.array): Feature matrix 
      y (numpy.array): Target vector
    """
    print("Normal equation:")
    w = jnp.linalg.inv(x.T @ x) @ x.T @ y
    loss = loss_fn(w, x, y)
    loss_neq = loss
    print("Weight: ", w)
    print("Loss: ", loss)

    return w, loss

print("Time: ", timeit(lambda: normal_eq(x1, y), number=1), "seconds")
with jax.disable_jit():
    print("Time(not-jit): ", timeit(lambda: normal_eq(x1, y), number=1), "seconds")
print("===============================================================")


# %%
# Gradient Descent

losses_gd = []

# Algorithm
# L := [(y_pred - y)^2] / n
# w := w + learning_rate * partial_L/partial_w
def steepest_gd(x, y):
    """
    Solve using steepest gradient descent

    Args:
      x (numpy.array): Feature matrix 
      y (numpy.array): Target vector
    """
    print("Steepest Gradient Descent:")
    w = random.uniform(key, (x.shape[1],), minval = 0, maxval = 1)
    for i in range(1000):
        w = update(w, x, y, lr=1e-2)
        losses_gd.append(loss_fn(w, x, y))

    y_pred = x @ w
    loss = jnp.mean((y - y_pred) ** 2)
    print("Weight: ", w)
    print("Loss: ", loss)

    return w, loss

print("Time: ", timeit(lambda: steepest_gd(x1, y), number=1), "seconds")
with jax.disable_jit():
    print("Time(not-jit): ", timeit(lambda: steepest_gd(x1, y), number=1), "seconds")
print("===============================================================")

# %%
# Stochastic Gradient Descent

losses_sgd = []

def stochastic_gd(x, y):
    """
    Solve using stochastic gradient descent

    Args:
      x (numpy.array): Feature matrix 
      y (numpy.array): Target vector
    """
    print("Stochastic Gradient Descent:")
    w = random.uniform(key, (x.shape[1],), minval = 0, maxval = 1)
    indices = random.choice(key, jnp.arange(x.shape[0]), (x.shape[0],), replace=False)
    for index in indices:
        w = update(w, x[index], y[index], lr=1e-4)
        losses_sgd.append(loss_fn(w, x, y))

    y_pred = x @ w
    loss = jnp.mean((y - y_pred) ** 2)
    print("Weight: ", w)
    print("Loss: ", loss)
    return w, loss

print("Time: ", timeit(lambda: stochastic_gd(x1, y), number=1), "seconds")
with jax.disable_jit():
    print("Time(not-jit): ", timeit(lambda: stochastic_gd(x1, y), number=1), "seconds")
print("===============================================================")

# %%
# Batch Gradient Descent

losses_bgd = []

def batch_gd(x, y):
    print("Batch Gradient Descent:")
    BATCH_SIZE = 200

    w = random.uniform(key, (x.shape[1],), minval = 0, maxval = 1)
    indices = random.choice(key, jnp.arange(x.shape[0]), (BATCH_SIZE,), replace=False)
    for _ in range(1000):
        w = update(w, x[indices, :], y[indices], lr=1e-4)
        losses_bgd.append(loss_fn(w, x, y))

    y_pred = x @ w
    loss = jnp.mean((y - y_pred) ** 2)
    print("Weight: ", w)
    print("Loss: ", loss)
    return w, loss

print("Time: ", timeit(lambda: batch_gd(x1, y), number=1), "seconds")
with jax.disable_jit():
    print("Time(not-jit): ", timeit(lambda: batch_gd(x1, y), number=1), "seconds")
print("===============================================================")

# %%
# FFN (Feedforward Neural Network)

def ffn(x, y, layers = [(20, nn.relu), (10, nn.sigmoid), (1, lambda x : x)]):
    input_size = 1 if len(x.shape) == 1 else x.shape[1]
    param_size = [layers[i-1][0] * layers[i][0] for i in range(1, len(layers))]
    w = random.uniform(key, (input_size, *[l[0] for l in layers]), dtype=jnp.float32)
    return w, param_size
    



# %%
# Plot

# plt.clf()
# plt.title("Training Loss")
# plt.xlabel("Iterations")
# plt.ylabel("Mean square error loss")
# plt.plot(list(range(1000)), [loss_neq for _ in range(1000)], label = "Normal Equation")
# plt.plot(list(range(1000)), losses_gd, label = "Steepest Gradient Descent")
# plt.plot(list(range(1000)), losses_sgd, label = "Stochastic Gradient Descent")
# plt.plot(list(range(1000)), losses_bgd, label = "Batch Gradient Descent")
# plt.legend()
# plt.show()

# %%

# %%

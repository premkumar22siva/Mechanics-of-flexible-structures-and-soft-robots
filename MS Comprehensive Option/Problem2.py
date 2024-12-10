import numpy as np
import matplotlib.pyplot as plt

################################
# Generate Data
################################

# Seed for reproducibility
np.random.seed(42)

# Generate 10 random x values within a range
x_generated = np.linspace(0, 5, 10)

# Parameters for the function (can use the previously fitted values or set randomly)
n_true = 0.06
a_true = 0.25
m_true = 0.57
b_true = 0.11

# Generate corresponding y values based on the function with added noise
noise = 0.001 * np.random.normal(0, 0.1, size=x_generated.shape)  # Add Gaussian noise
y_generated = n_true * np.exp(-a_true * (m_true * x_generated + b_true) ** 2) + noise

# Display the generated x and y arrays
x_generated, y_generated

##################################
# Compute Loss
##################################

def compute_loss(x, y, n, a, m, b):
  """
  Compute the Mean Squared Error (MSE) loss.

  Parameters:
  x : np.array
    Input data points (x values).
  y : np.array
    Actual output data points (y values).
  n, a, m, b : float
    Parameters of the function y = n * exp(-a * (m * x + b)^2).

  Returns:
    float
      Mean Squared Error (MSE) loss.
  """
  y_int = (m * x + b) ** 2
  y_pred = n * np.exp(-a * y_int)
  return np.mean((y - y_pred) ** 2)

##########################################
# Gradient Descent and Backpropagation
##########################################

# Generate data points directly as NumPy arrays (without pandas)
x_data = x_generated
y_data = y_generated

# Reinitialize parameters (n, a, m, b)
n_fit = np.random.rand()
a_fit = np.random.rand()
m_fit = np.random.rand()
b_fit = np.random.rand()

epochs = 10000
learning_rate = 0.001

# Perform gradient descent for the generated data
for epoch in range(epochs):
  # Forward pass: compute intermediate and final outputs
  y_int_fit = (m_fit * x_data + b_fit) ** 2
  y_pred_fit = n_fit * np.exp(-a_fit * y_int_fit)

  # Compute gradients
  # Gradients for n and a (output layer)
  grad_n_fit = -2 * np.mean((y_data - y_pred_fit) * np.exp(-a_fit * y_int_fit))
  grad_a_fit = -2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-y_int_fit))

  # Gradients for m and b (inner layer)
  grad_m_fit = -2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-a_fit) * (2 * (m_fit * x_data + b_fit) * x_data))
  grad_b_fit = -2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-a_fit) * (2 * (m_fit * x_data + b_fit)))

  # Update parameters
  n_fit -= learning_rate * grad_n_fit
  a_fit -= learning_rate * grad_a_fit
  m_fit -= learning_rate * grad_m_fit
  b_fit -= learning_rate * grad_b_fit

  # Compute loss for monitoring
  loss_fit = compute_loss(x_data, y_data, n_fit, a_fit, m_fit, b_fit)

  # Print loss every 100 epochs
  if epoch % 1000 == 0:
    print(f"Epoch {epoch}: Loss = {loss_fit:.6f}")

# Final fitted parameter values
print(f"\nValue of n after training: {n_fit: .18f}")
print(f"Value of a after training: {a_fit: .18f}")
print(f"Value of m after training: {m_fit: .18f}")
print(f"Value of b after training: {b_fit: .18f}")

##################################
# Visualization
##################################

# Predicted y values using the fitted parameters
y_predicted = n_fit * np.exp(-a_fit * (m_fit * x_generated + b_fit) ** 2)

# Plot the training data
plt.scatter(x_generated, y_generated, color='blue', label='Training Data (Noisy)', marker='o')

# Plot the predicted data
plt.plot(x_generated, y_predicted, color='red', label='Predicted Data (Model)', linestyle='-')

# Add labels, title, and legend
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Training Data and Predicted Data")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

###################################
# Sensitivity of Learning Rate
###################################

# Reinitialize parameters (n, a, m, b)
n_fit = np.random.rand()
a_fit = np.random.rand()
m_fit = np.random.rand()
b_fit = np.random.rand()

epochs = 10000
learning_rate = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])

n_final = np.zeros(len(learning_rate))
a_final = np.zeros(len(learning_rate))
m_final = np.zeros(len(learning_rate))
b_final = np.zeros(len(learning_rate))

# Perform gradient descent for the generated data
for i in range(len(learning_rate)):
  for epoch in range(epochs):
    # Forward pass: compute intermediate and final outputs
    y_int_fit = (m_fit * x_data + b_fit) ** 2
    y_pred_fit = n_fit * np.exp(-a_fit * y_int_fit)

    # Compute gradients
    # Gradients for n and a (output layer)
    grad_n_fit = -2 * np.mean((y_data - y_pred_fit) * np.exp(-a_fit * y_int_fit))
    grad_a_fit = -2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-y_int_fit))

    # Gradients for m and b (inner layer)
    grad_m_fit = -2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-a_fit) * (2 * (m_fit * x_data + b_fit) * x_data))
    grad_b_fit = -2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-a_fit) * (2 * (m_fit * x_data + b_fit)))

    # Update parameters
    n_fit -= learning_rate[i] * grad_n_fit
    a_fit -= learning_rate[i] * grad_a_fit
    m_fit -= learning_rate[i] * grad_m_fit
    b_fit -= learning_rate[i] * grad_b_fit

  n_final[i] = n_fit
  a_final[i] = a_fit
  m_final[i] = m_fit
  b_final[i] = b_fit

  n_fit = np.random.rand()
  a_fit = np.random.rand()
  m_fit = np.random.rand()
  b_fit = np.random.rand()

plt.figure(2)
plt.scatter(x_generated, y_generated)
for i in range(len(learning_rate)):
  y_predicted = n_final[i] * np.exp(-a_final[i] * (m_final[i] * x_generated + b_final[i]) ** 2)
  plt.plot(x_generated, y_predicted, label=f"{learning_rate[i]}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend(title = "Learning Rate", loc = "upper right")
plt.title("Predicted values for different Learning Rates")
plt.show()

#################################
# Sensitivity of epochs
#################################

# Reinitialize parameters (n, a, m, b)
n_fit = np.random.rand()
a_fit = np.random.rand()
m_fit = np.random.rand()
b_fit = np.random.rand()

epochs = np.array([100000, 10000, 1000, 100, 10])
learning_rate = 0.001

n_final = np.zeros(len(epochs))
a_final = np.zeros(len(epochs))
m_final = np.zeros(len(epochs))
b_final = np.zeros(len(epochs))

# Perform gradient descent for the generated data
for i in range(len(epochs)):
  for epoch in range(epochs[i]):
    # Forward pass: compute intermediate and final outputs
    y_int_fit = (m_fit * x_data + b_fit) ** 2
    y_pred_fit = n_fit * np.exp(-a_fit * y_int_fit)

    # Compute gradients
    # Gradients for n and a (output layer)
    grad_n_fit = -2 * np.mean((y_data - y_pred_fit) * np.exp(-a_fit * y_int_fit))
    grad_a_fit = -2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-y_int_fit))

    # Gradients for m and b (inner layer)
    grad_m_fit = -2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-a_fit) * (2 * (m_fit * x_data + b_fit) * x_data))
    grad_b_fit = -2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-a_fit) * (2 * (m_fit * x_data + b_fit)))

    # Update parameters
    n_fit -= learning_rate * grad_n_fit
    a_fit -= learning_rate * grad_a_fit
    m_fit -= learning_rate * grad_m_fit
    b_fit -= learning_rate * grad_b_fit

  n_final[i] = n_fit
  a_final[i] = a_fit
  m_final[i] = m_fit
  b_final[i] = b_fit

  n_fit = np.random.rand()
  a_fit = np.random.rand()
  m_fit = np.random.rand()
  b_fit = np.random.rand()

plt.figure(3)
plt.scatter(x_generated, y_generated)
for i in range(len(epochs)):
  y_predicted = n_final[i] * np.exp(-a_final[i] * (m_final[i] * x_generated + b_final[i]) ** 2)
  plt.plot(x_generated, y_predicted, label=f"{epochs[i]}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend(title=" epochs", loc = "upper right")
plt.title("Predicted values for different epochs")
plt.show()

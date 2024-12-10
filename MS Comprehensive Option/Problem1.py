import numpy as np
import matplotlib.pyplot as plt

###############################
# Generate data
###############################

# Seed for reproducibility
np.random.seed(42)
# Generating 10 random x values within a range
x_generated = np.linspace(0, 5, 10)

# Parameters for the function (can use the previously fitted values or set randomly)
n_true = 0.06
a_true = 0.25
m_true = 0.57
b_true = 0.11

# Generating corresponding y values based on the function with added noise
noise = 0.001 * np.random.normal(0, 0.1, size = x_generated.shape)
y_generated = n_true * np.exp( - a_true * (m_true * x_generated + b_true) ** 2.0 ) + noise

# Display the generated x and y arrays
x_generated, y_generated

####################################
# Compute Loss
####################################

def compute_loss(x, y, m, b):

  y_pred = m * x + b
  loss = np.mean( (y - y_pred) ** 2.0 )

  return loss

###################################
# Model
###################################

x_data = x_generated
y_data = y_generated

# Initialize parameters (m, b)
m_fit = np.random.rand()
b_fit = np.random.rand()

epochs =  10000
learning_rate = 0.001

for epoch in range(epochs):
  # Forward
  y_pred_fit = m_fit * x_data + b_fit

  # Compute gradients
  grad_m_fit = -2.0 * np.mean( x_data * (y_data - y_pred_fit) )
  grad_b_fit = -2.0 * np.mean( (y_data - y_pred_fit) )

  # Update
  m_fit -= learning_rate * grad_m_fit
  b_fit -= learning_rate * grad_b_fit

  if epoch % 1000 == 0:
    loss_fit = compute_loss(x_data, y_data, m_fit, b_fit)
    print(f"Epoch: {epoch}, Loss: {loss_fit}")

# Display
print(f"\nValue of m after training: {m_fit: .18f}")
print(f"Value of b after training: {b_fit: .18f}")

#######################################
# Visualize
#######################################

y_predicted = m_fit * x_generated + b_fit

plt.scatter(x_generated, y_generated)
plt.plot(x_generated, y_predicted, color = "red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Training data (noisy)", "Predicted data (model)"])
plt.show()

#######################################
# Sensitivity of Learning Rate
#######################################

'''Fixing the epochs to 10000 and varying learning rate'''

# Initialize parameters (m, b)
m_fit = np.random.rand()
b_fit = np.random.rand()

epochs =  10000
learning_rate = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])

m_final = np.zeros(len(learning_rate))
b_final = np.zeros(len(learning_rate))

for i in range(len(learning_rate)):

  for epoch in range(epochs):
    # Forward
    y_pred_fit = m_fit * x_data + b_fit

    # Compute gradients
    grad_m_fit = -2.0 * np.mean( x_data * (y_data - y_pred_fit) )
    grad_b_fit = -2.0 * np.mean( (y_data - y_pred_fit) )

    # Update
    m_fit -= learning_rate[i] * grad_m_fit
    b_fit -= learning_rate[i] * grad_b_fit

  m_final[i] = m_fit
  b_final[i] = b_fit

  m_fit = np.random.rand()
  b_fit = np.random.rand()

plt.figure(2)
plt.scatter(x_generated, y_generated)
for i in range(len(learning_rate)):
  y_predicted = m_final[i] * x_generated + b_final[i]
  plt.plot(x_generated, y_predicted, label=f"{learning_rate[i]}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend(title = "Learning Rate", loc = "upper right")
plt.title("Predicted values for different Learning Rates")
plt.show()

#############################################
# Sensitivity of epochs
#############################################

'''Fixing the learning rate to 0.001 and varying epochs'''

# Initialize parameters (m, b)
m_fit = np.random.rand()
b_fit = np.random.rand()

epochs =  np.array([100000, 10000, 1000, 100, 10])
learning_rate = 0.001

m_final = np.zeros(len(epochs))
b_final = np.zeros(len(epochs))

for i in range(len(epochs)):

  for epoch in range(epochs[i]):
    # Forward
    y_pred_fit = m_fit * x_data + b_fit

    # Compute gradients
    grad_m_fit = -2.0 * np.mean( x_data * (y_data - y_pred_fit) )
    grad_b_fit = -2.0 * np.mean( (y_data - y_pred_fit) )

    # Update
    m_fit -= learning_rate * grad_m_fit
    b_fit -= learning_rate * grad_b_fit


  m_final[i] = m_fit
  b_final[i] = b_fit

  m_fit = np.random.rand()
  b_fit = np.random.rand()

plt.figure(3)
plt.scatter(x_generated, y_generated)
for i in range(len(epochs)):
  y_predicted = m_final[i] * x_generated + b_final[i]
  plt.plot(x_generated, y_predicted, label=f"{epochs[i]}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend(title=" epochs", loc = "upper right")
plt.title("Predicted values for different epochs")
plt.show()

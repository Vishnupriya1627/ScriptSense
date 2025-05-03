import torch

def sigmoid_like_transform(x):
    # Shift and scale the sigmoid to control shape and range
    # Adjust parameters for smooth curvature
    steepness = 15
    center = 0.875  # midpoint of steep curve between 0.8 and 0.95

    # Sigmoid function scaled to output range [0.5, 1]
    y = 0.5 + 0.5 * torch.sigmoid(steepness * (x - center))
    return y

import matplotlib.pyplot as plt

x_vals = torch.linspace(0, 1, 300)
y_vals = sigmoid_like_transform(x_vals)

plt.plot(x_vals.numpy(), y_vals.numpy(), label='Transformed Score', color='blue')
plt.xlabel("Original Score")
plt.ylabel("Transformed Score")
plt.grid(True)
plt.legend()
plt.show()


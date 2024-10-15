
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Provided data
data_list = [
    (0.749, 6.334), (1.901, 9.405), (1.464, 8.484), (1.197, 5.604), (0.312, 4.716), (0.312, 5.293),
    (0.116, 5.826), (1.732, 8.679), (1.202, 6.798), (1.416, 7.747), (0.041, 5.039), (1.940, 10.148),
    (1.665, 8.465), (0.425, 5.787), (0.364, 5.188), (0.367, 6.069), (0.608, 5.123), (1.050, 6.821),
    (0.864, 6.200), (0.582, 4.284), (1.224, 7.967), (0.279, 5.098), (0.584, 5.758), (0.733, 5.964),
    (0.912, 5.321), (1.570, 8.290), (0.399, 4.855), (1.028, 6.283), (1.185, 7.393), (0.093, 4.683),
    (1.215, 9.531), (0.341, 5.198), (0.130, 4.648), (1.898, 9.619), (1.931, 7.875), (1.617, 8.824),
    (0.609, 5.888), (0.195, 7.049), (1.368, 7.913), (0.880, 6.942), (0.244, 4.698), (0.990, 5.802),
    (0.069, 5.349), (1.819, 10.208), (0.518, 6.344), (1.325, 7.066), (0.623, 7.273), (1.040, 5.719),
    (1.093, 7.867), (0.370, 7.300), (1.939, 8.827), (1.550, 8.084), (1.879, 9.737), (1.790, 8.865),
    (1.196, 6.037), (1.844, 9.600), (0.177, 3.469), (0.392, 5.649), (0.090, 3.352), (0.651, 7.502),
    (0.777, 5.549), (0.543, 5.306), (1.657, 9.786), (0.714, 4.910), (0.562, 5.913), (1.085, 8.563),
    (0.282, 3.238), (1.604, 8.998), (0.149, 4.707), (1.974, 10.703), (1.544, 7.397), (0.397, 3.872),
    (0.011, 4.555), (1.631, 9.190), (1.414, 8.492), (1.458, 8.720), (1.543, 7.948), (0.148, 4.677),
    (0.717, 6.444), (0.232, 3.981), (1.726, 11.044), (1.247, 8.214), (0.662, 4.794), (0.127, 5.038),
    (0.622, 4.891), (0.650, 6.738), (1.459, 9.536), (1.275, 7.005), (1.774, 10.287), (0.944, 7.246),
    (0.239, 5.540), (1.426, 10.176), (1.522, 8.319), (1.123, 6.614), (1.542, 7.736), (0.988, 6.147),
    (1.045, 7.059), (0.855, 6.906), (0.051, 4.429), (0.216, 5.475)
]

# Convert to DataFrame
data = pd.DataFrame(data_list, columns=['X', 'Y'])

# Perform the covariance-based linear regression
X = data['X'].values
Y = data['Y'].values

# Calculate means of X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Calculate the covariance of X and Y and variance of X
cov_xy = np.sum((X - mean_x) * (Y - mean_y))
var_x = np.sum((X - mean_x) ** 2)

# Calculate the slope and intercept
slope = cov_xy / var_x
intercept = mean_y - slope * mean_x

# Print the linear model
print(f'Linear model: Y = {slope:.3f} * X + {intercept:.3f}')

# Plot the data points
plt.scatter(X, Y, color='blue', label='Data points')

# Plot the regression line
y_pred = slope * X + intercept
plt.plot(X, y_pred, color='red', label='Regression line')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')

# Show legend
plt.legend()

# Display the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Generate x values from -10 to 10
x = np.linspace(0, 10, 100)

# Calculate corresponding y values for y = x
y = x

# Plot the function
plt.plot(x, y, label='y = x')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = x')

# Add a legend
plt.legend()

# Show the plot
plt.show()
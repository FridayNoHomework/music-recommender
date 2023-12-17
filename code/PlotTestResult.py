import pickle
with open('../data/serialized_data/test_result.pkl', 'rb') as file:
    result_dict = pickle.load(file)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data
data1 = np.array(result_dict['benchmark1_num'])
data2 = np.array(result_dict['benchmark2_ti'])
data3 = np.array(result_dict['w2v'])

# Custom x and y values
x = np.array([2, 4, 6, 8])  # Custom x-axis values
y = np.array([100, 150, 300])  # Custom y-axis values

xx, yy = np.meshgrid(x, y)

# Flatten the DataFrames (convert to NumPy arrays) and use them as z values
z1 = data1.flatten()
z2 = data2.flatten()
z3 = data3.flatten()

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))  # Adjust figure size
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface with soft, solid colors and legend
ax.plot_surface(xx, yy, z1.reshape(xx.shape), color='lightblue', label='Numerical only (bm1)')
ax.plot_surface(xx, yy, z2.reshape(xx.shape), color='lightcoral', label='TF-IDF (bm2)')
ax.plot_surface(xx, yy, z3.reshape(xx.shape), color='lightgreen', label='Word2Vec w/ user data')

# Set labels and legend
ax.set_xlabel('# of songs used to generate taste vector')
ax.set_ylabel('K (# of songs RS recommends)')
ax.set_zlabel('Hit@K')
ax.set_title('Hit@K for different models')

# Add legend with improved formatting
ax.legend(loc='upper right', fancybox=True, title='Models')

plt.show()
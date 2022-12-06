import numpy as np
from matplotlib import pyplot as plt


def get_obsticle_ocupancy(radius):
    x = np.arange(0, radius*2+3, 1)
    y = np.arange(0, radius*2+3, 1)
    X, Y = np.meshgrid(x, y)
    distances = np.sqrt((X - radius)**2 + (Y - radius)**2)
    binary_map = distances <= radius
    binary_map = binary_map.astype(int)
    occupancy_xy = np.array(np.where(binary_map == 1)).T
    return occupancy_xy


grid = np.zeros((40, 40))
radius = 5
binary_map = get_obsticle_ocupancy(radius)
obstacle_center = np.array([10,10])
center = obstacle_center + radius
occupancy = binary_map + center
grid[occupancy[:,0], occupancy[:,1]] = 1

print(grid)
plt.imshow(grid, cmap='gray')
plt.show()

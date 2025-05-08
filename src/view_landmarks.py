import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
def load_world_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split()
            landmark_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            data.append([landmark_id, x, y, z])
    return np.array(data)

# Plot the 3D landmarks
def plot_landmarks(world_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = world_data[:, 1]
    ys = world_data[:, 2]
    zs = world_data[:, 3]

    ax.scatter(xs, ys, zs, c='b', marker='o')
    # for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
    #     ax.text(x, y, z, f'{int(world_data[i, 0])}', size=8, zorder=1, color='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Landmarks from world.dat')
    plt.tight_layout()
    plt.show()

# Example usage
world_data = load_world_data('../data/world.dat')
plot_landmarks(world_data)

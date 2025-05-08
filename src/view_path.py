import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_paths(filename):
    odometry_path = []
    groundtruth_path = []
    with open(filename, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue  # Skip if not enough data
            # First 3 values and last 3 values
            x1, y1, z1 = map(float, parts[1:4])  # skip index
            x2, y2, z2 = map(float, parts[-3:])
            odometry_path.append([x1, y1, z1])
            groundtruth_path.append([x2, y2, z2])
    return np.array(odometry_path), np.array(groundtruth_path)

def plot_two_paths(path1, path2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(path1[:, 0], path1[:, 1], path1[:, 2], label='Odometry Predicted Path', color='orange')
    ax.plot(path2[:, 0], path2[:, 1], path2[:, 2], label='Ground Truth', color='blue')
    
    # Mark start and end points
    ax.scatter(path1[0, 0], path1[0, 1], path1[0, 2], color='green', s=50, label='Start')
    ax.scatter(path2[0, 0], path2[0, 1], path2[0, 2], color='green', s=50)
    ax.scatter(path1[-1, 0], path1[-1, 1], path1[-1, 2], color='red', s=50, label='End')
    ax.scatter(path2[-1, 0], path2[-1, 1], path2[-1, 2], color='red', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Comparison of Initial and Final Paths')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Example usage
initial_path, final_path = load_paths('../data/trajectory.dat')  # Replace with your filename
plot_two_paths(initial_path, final_path)

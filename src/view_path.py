import numpy as np
import matplotlib.pyplot as plt

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
            x1, y1, theta1 = map(float, parts[1:4])  # skip index
            x2, y2, theta2 = map(float, parts[-3:])
            odometry_path.append([x1, y1, theta1])
            groundtruth_path.append([x2, y2, theta2])
    return np.array(odometry_path), np.array(groundtruth_path)

def load_single_path(filename):
    path = []
    with open(filename, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            x, y, theta = map(float, parts[:3])  # skip index
            path.append([x, y, theta])
    return np.array(path)

def plot_three_paths_with_orientation(path1, path2, path3):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot paths
    ax.plot(path1[:, 0], path1[:, 1], label='Odometry Predicted Path', color='orange')
    ax.plot(path2[:, 0], path2[:, 1], label='Ground Truth', color='blue')
    ax.plot(path3[:, 0], path3[:, 1], label='Kalman Estimated Path', color='green')

    # Orientation arrows
    scale = 0.3
    ax.quiver(path1[:, 0], path1[:, 1], np.cos(path1[:, 2]), np.sin(path1[:, 2]),
              color='orange', scale_units='xy', angles='xy', scale=1, width=0.003)
    ax.quiver(path2[:, 0], path2[:, 1], np.cos(path2[:, 2]), np.sin(path2[:, 2]),
              color='blue', scale_units='xy', angles='xy', scale=1, width=0.003)
    ax.quiver(path3[:, 0], path3[:, 1], np.cos(path3[:, 2]), np.sin(path3[:, 2]),
              color='green', scale_units='xy', angles='xy', scale=1, width=0.003)

    # Start and end points
    for path in [path1, path2, path3]:
        ax.scatter(path[0, 0], path[0, 1], color='black', s=50, label='Start' if path is path1 else "")
        ax.scatter(path[-1, 0], path[-1, 1], color='red', s=50, label='End' if path is path1 else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Comparison of Odometry, Ground Truth, and Kalman Paths')
    ax.axis('equal')
    ax.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Load paths
odometry_path, groundtruth_path = load_paths('../data/trajectory.dat')
kalman_path = load_single_path('../output/kalman_path.txt')

# Plot all
plot_three_paths_with_orientation(odometry_path, groundtruth_path, kalman_path)
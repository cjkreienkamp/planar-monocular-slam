import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def euler_to_theta(rot_x, rot_y, rot_z):
    """
    Convert Euler angles to a single theta value (rotation around Z-axis).
    Since this is a planar robot, we only care about rotation around Z.
    """
    return rot_z

def load_optimized_poses(filename):
    poses = []
    thetas = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            trans_x, trans_y, trans_z, rot_x, rot_y, rot_z = map(float, line.strip().split())
            poses.append([trans_x, trans_y, trans_z])
            theta = euler_to_theta(rot_x, rot_y, rot_z)
            thetas.append(theta)
            print(f"Camera {i}: pos=({trans_x:.2f}, {trans_y:.2f}, {trans_z:.2f}), theta={theta:.2f}")
    return np.array(poses), np.array(thetas)

def load_optimized_points(filename):
    points = {}
    with open(filename, 'r') as f:
        for line in f:
            id_, x, y, z = line.strip().split()
            points[int(id_)] = [float(x), float(y), float(z)]
    return points

def visualize_results(poses_file, points_file):
    # Load data
    poses, thetas = load_optimized_poses(poses_file)
    points = load_optimized_points(points_file)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera path
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], 'b-', label='Camera Path')
    ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], c='b', marker='o')
    
    # Plot camera orientations
    for i in range(len(poses)):
        # Draw a line indicating camera orientation
        direction = np.array([np.cos(thetas[i]), np.sin(thetas[i]), 0]) * 0.5
        ax.quiver(poses[i, 0], poses[i, 1], poses[i, 2],
                 direction[0], direction[1], direction[2],
                 color='g', alpha=0.5)
    
    # Plot 3D points
    points_array = np.array(list(points.values()))
    ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
              c='r', marker='.', label='Landmarks')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Optimized Camera Path and 3D Points\n(Camera indices: 0-198)')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        points_array[:, 0].max() - points_array[:, 0].min(),
        points_array[:, 1].max() - points_array[:, 1].min(),
        points_array[:, 2].max() - points_array[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points_array[:, 0].max() + points_array[:, 0].min()) * 0.5
    mid_y = (points_array[:, 1].max() + points_array[:, 1].min()) * 0.5
    mid_z = (points_array[:, 2].max() + points_array[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

if __name__ == '__main__':
    visualize_results('../output/optimized_poses.txt', '../output/optimized_points.txt') 
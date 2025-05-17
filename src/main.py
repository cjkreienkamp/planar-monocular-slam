import math
import numpy as np
import os
import cv2

def add_new_landmarks(mu, sigma, measurements, id_to_state_map):
    
    #robot
    robot_xy = mu[0:2] #translational part of the robot pose
    robot_theta = mu[2] #rotation of the robot
    c = math.cos(robot_theta[0])
    s = math.sin(robot_theta[0])
    R_r2w = np.array([[c, -s], [s, c]])
    
    #landmarks
    n_landmarks = int((len(mu)-3)/2)
    
    # add NEW landmarks without applying any correction
    for measurement in measurements:
        
        state_pos_of_landmark = id_to_state_map[ measurement['id'] ] 
        
        if state_pos_of_landmark == -1: # new landmark
            n_landmarks += 1
            id_to_state_map[ measurement['id'] ] = n_landmarks
        
            # add the landmark position to the full state vector 
            lm_wrt_robot = np.array([[measurement['x_pose']], [measurement['y_pose']]])
            lm_wrt_world = robot_xy + R_r2w @ lm_wrt_robot
            mu = np.concatenate([mu, lm_wrt_world], axis=0)

            # adding the landmark covariance to the full covariance
            large_initial_uncertainty = 2
            sigma = np.block([
                [sigma,                         np.zeros((sigma.shape[0], 2))],
                [np.zeros((2, sigma.shape[1])), large_initial_uncertainty * np.eye(2)]
            ])
        break #TEST

    return mu, sigma, id_to_state_map

def prediction(mu, sigma, control_input):
    # get control inputs from old and new odometry poses
    # use mu and control inputs to predict x_t = f(x_t-1, u_t-1)

    mu_x, mu_y, robot_theta = mu[:3,0]
    u1, u2 = control_input

    # Jacobian A = df(x,u)/dx
    A = np.eye(len(mu))
    A[0,2] = -u1*np.sin(robot_theta)
    A[1,2] = u1*np.cos(robot_theta)

    # Jacobian B = df(x,u)/du
    u_dim = 2
    B = np.zeros((len(mu), u_dim))
    B[0,0] = np.cos(robot_theta)
    B[1,0] = np.sin(robot_theta)
    B[2,1] = 1

    # predict new robot state through transition function f(x,u)
    mu[0] = mu_x + u1 * np.cos(robot_theta)
    mu[1] = mu_y + u1 * np.sin(robot_theta)
    mu[2] = robot_theta + u2
    
    # predict new state covariance due to control noise
    sigma_u = 0.1 #constant part
    sigma_u = np.array([
        [sigma_u**2 + u1**2, 0],
        [ 0, sigma_u**2 + u2**2]
        ])
    sigma = A @ sigma @ A.T + B @ sigma_u @ B.T

    return mu, sigma

def correction(mu, sigma, measurements, id_to_state_map):
    
    if len(measurements) == 0:
        return mu, sigma

    # robot
    robot_xy = mu[0:2] #translational part of the robot pose
    robot_theta = mu[2] #rotation of the robot
    c = math.cos(robot_theta[0])
    s = math.sin(robot_theta[0])
    R = np.array([[c, -s], [s, c]])
    Rt = np.array([[c, s], [-s, c]])
    Rtp = np.array([[-s, c], [-c, -s]]) #derivative of transposed rotation matrix

    # landmarks
    z = np.empty((0,1))
    h = np.empty((0,1))
    C = np.empty((0,len(mu)))
    n_known_landmarks = 0;
    for measurement in measurements:
    
        #fetch the position in the state vector corresponding to the actual measurement
        state_pos_of_landmark = id_to_state_map[ measurement['id'] ] 

        # if current landmark is a REOBSERVED landmark 
        if state_pos_of_landmark != -1:
            n_known_landmarks += 1
            idx_landmark_x = 3 + 2*( state_pos_of_landmark - 1 )

            actual_measurement = np.array([measurement['x_pose'], measurement['y_pose']])
            z = np.concatenate([z, actual_measurement.reshape(2,1)], axis=0)
            
            # lm_wrt_world = robot_xy + R @ lm_wrt_robot
            lm_wrt_world = mu[idx_landmark_x:idx_landmark_x+2]
            predicted_measurement = Rt @ (lm_wrt_world - robot_xy)
            h = np.concatenate([h, predicted_measurement.reshape(2,1)], axis=0)

            C_lm = np.zeros((2, len(mu)));
            C_lm[:2,:2] = -Rt
            C_lm[:2,2] = (Rtp @ (lm_wrt_world - robot_xy)).flatten()
            C_lm[:,idx_landmark_x:idx_landmark_x+2] = Rt
            C = np.concatenate([C, C_lm], axis=0)

    #if I have seen again at least one landmark
    #I need to update, otherwise I jump to the new landmark case
    if n_known_landmarks > 0:
        noise = 0.01
        sigma_z = np.eye(2 * n_known_landmarks) * noise
        K = sigma @ C.T @ np.linalg.inv(sigma_z + C @ sigma @ C.T)
        innovation = z - h
        mu = mu + 0.001*(K @ innovation).reshape(-1, 1) # remove 0.01*
        sigma = (np.eye(len(mu)) - K @ C) @ sigma

    return mu, sigma

def load_file(filepath):
    measurements = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("point"):
                tokens = line.strip().split()
                # tokens[0] = 'point', tokens[1] = internal index, tokens[2:] = ID, x, y
                measurement = {
                    "id": int(tokens[2]),
                    "x_pose": float(tokens[3]),
                    "y_pose": float(tokens[4])
                    }
                measurements.append(measurement)
            elif line.startswith("gt_pose:"):
                tokens = line.strip().split()
                #tokens[0] = 'gt_pose:', tokens[1:] = x, y, z
                pose = np.array([float(text) for text in tokens[1:]]).reshape(3,1)
    return pose, measurements

def get_matching_pts(filepath1, filepath2):
    frame1_ids, matching_ids = [], []
    frame1_pts_all, frame2_pts = [], []
    with open(filepath1, 'r') as file:
        for line in file:
            if line.startswith("point"):
                tokens = line.strip().split()
                # tokens[0] = 'point', tokens[1] = internal index, tokens[2:] = ID, x, y
                frame1_ids.append(int(tokens[2]))
                frame1_pts_all.append([float(tokens[3]), float(tokens[4])])
    
    with open(filepath2, 'r') as file:
        for line in file:
            if line.startswith("point"):
                tokens = line.strip().split()
                # tokens[0] = 'point', tokens[1] = internal index, tokens[2:] = ID, x, y
                if int(tokens[2]) in frame1_ids:
                    matching_ids.append(int(tokens[2]))
                    frame2_pts.append([float(tokens[3]), float(tokens[4])])
    
    frame1_pts = np.array([pt for id, pt in zip(frame1_ids, frame1_pts_all) if id in matching_ids])
    frame2_pts = np.array(frame2_pts)
    
    return frame1_pts, frame2_pts, matching_ids

def get_odometry(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("odom_pose:"):
                tokens = line.strip().split()
                # tokens[0] = 'odom_pose:', tokens[1] = x, tokens[2] = y, tokens[3] = theta
                x = float(tokens[1])
                y = float(tokens[2])
                theta = float(tokens[3])
                return x, y, theta

def odometry2transform(x, y, theta):
    T = np.eye(4)
    T[0, 0] =  np.cos(theta)
    T[0, 1] = -np.sin(theta)
    T[1, 0] =  np.sin(theta)
    T[1, 1] =  np.cos(theta)
    T[0, 3] = x
    T[1, 3] = y
    return T

def compute_projection_matrix(R_pixel_camera , T_world_robot, T_robot_camera):
    # projection matrix maps 3D --> 2D <==> TO pixel FROM world
    T_camera_world = np.linalg.inv(T_world_robot @ T_robot_camera)
    R_camera_world = T_camera_world[:3, :3]
    t_camera_world = T_camera_world[:3, 3].reshape(3, 1)
    extrinsic = np.hstack((R_camera_world, t_camera_world))  # 3x4
    intrinsic = R_pixel_camera
    return intrinsic @ extrinsic 

def compute_6dof_pose(T_robot_camera, T_rw):
    T_cw = T_robot_camera @ T_rw
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    rvec = R.from_matrix(R_cw).as_rotvec()
    pose = np.concatenate([rvec, t_cw])  # 6 parameters
    return pose

if __name__ == '__main__':
    from scipy.spatial.transform import Rotation as R
    intrinsic = np.array([  [180,   0, 320],
                            [  0, 180, 240],
                            [  0,   0,   1]])
    # T_<to>_<from>
    T_robot_camera = np.array([ [ 0,  0, 1, 0.2],
                                [-1,  0, 0,   0],
                                [ 0, -1, 0,   0],
                                [ 0,  0, 0,   1]])
    
    files = [f"../data/meas-00{num:03d}.dat" for num in range(0,200)]

    camera_poses = []
    points_3d = {}  # point_id -> np.array([x, y, z])
    observations = []  # list of (cam_idx, point_id, x, y)

    for cam_idx in range(len(files)-1):
        filepath1 = files[cam_idx]
        filepath2 = files[cam_idx+1]

        x1, y1, theta1 = get_odometry(filepath1)
        x2, y2, theta2 = get_odometry(filepath2)
        T_world_robot1 = odometry2transform(x1, y1, theta1)
        T_world_robot2 = odometry2transform(x2, y2, theta2)
        P1 = compute_projection_matrix(intrinsic, T_world_robot1, T_robot_camera)
        P2 = compute_projection_matrix(intrinsic, T_world_robot2, T_robot_camera)

        pts1, pts2, matching_ids = get_matching_pts(filepath1, filepath2)
        pts_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts_3d = (pts_4d_hom / pts_4d_hom[3])[:3].T  # Nx3

        if cam_idx == 0:
            camera_poses.append(compute_6dof_pose(T_robot_camera, T_world_robot1))
        camera_poses.append(compute_6dof_pose(T_robot_camera, T_world_robot2))

        for id, p3d, pt1, pt2 in zip(matching_ids, pts_3d, pts1, pts2):
            if id not in points_3d:
                points_3d[id] = p3d  # Save only the first triangulation of each landmark
            observations.append((cam_idx, id, pt1[0], pt1[1]))
            observations.append((cam_idx + 1, id, pt2[0], pt2[1]))

    # Save to file
    with open("../output/ba_input.txt", "w") as f:
        f.write(f"{len(camera_poses)} {len(points_3d)} {len(observations)}\n")
        for cam_idx, pid, x, y in observations:
            f.write(f"{cam_idx} {pid} {x} {y}\n")
        for pose in camera_poses:
            f.write(" ".join(map(str, pose)) + "\n")
        for pid in sorted(points_3d.keys()):
            f.write(f"{pid} " + " ".join(map(str, points_3d[pid])) + "\n")



def old_kalman():
    # estimate points in the world from image correspondences + odometry data
    # use correspondences from all images to find world points that minimize the distance among all features directions
    # correspondences known,

    sigma = np.array([ [1,0,0], [0,1,0], [0,0,1] ])
    id_to_state_map = 1000*[-1]

    filepath = "../data/meas-00000.dat"
    odometry, sensor_measurements = load_file(filepath)
    print(sensor_measurements)
    quit()

    mu, sigma, id_to_state_map = add_new_landmarks(odometry, sigma, sensor_measurements, id_to_state_map)
    prev_odometry = odometry

    i = 0
    kalman_path = [f"{mu[0][0]} {mu[1][0]} {mu[2][0]}\n"]
    files = [f"../data/meas-00{num:03d}.dat" for num in range(1,200)]
    for filepath in files:
        print(mu)
        print(sigma)
        print()
        
        odometry, sensor_measurements = load_file(filepath)

        u1 = np.linalg.norm( np.array(odometry[:2]) - np.array(prev_odometry[:2]) )
        u2 = (odometry[2] - prev_odometry[2] )[0]
        mu, sigma = prediction(mu, sigma, [u1,u2])

        # mu, sigma = correction(mu, sigma, sensor_measurements, id_to_state_map)
        kalman_path.append(f"{mu[0][0]} {mu[1][0]} {mu[2][0]}\n")
        
        #TEST mu, sigma, id_to_state_map = add_new_landmarks(mu, sigma, sensor_measurements, id_to_state_map)
        prev_odometry = odometry

    with open("../output/kalman_path.txt","w") as f:
        f.writelines(kalman_path)

    # Questions:
    #   * Where is their repository?
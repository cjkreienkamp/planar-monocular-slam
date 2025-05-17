#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <map>

using namespace std;
using namespace g2o;

int main() {
    ifstream infile("/home/chris/sapienza/probabilistic-robotics/03-PlanarMonocularSLAM/output/ba_input.txt");
    if (!infile.is_open()) {
        cerr << "Failed to open input file" << endl;
        return -1;
    }

    int num_cams, num_points, num_obs;
    infile >> num_cams >> num_points >> num_obs;

    struct Obs {
        int cam_idx, pt_idx;
        float u, v;
    };
    vector<Obs> observations;
    for (int i = 0; i < num_obs; ++i) {
        Obs obs;
        infile >> obs.cam_idx >> obs.pt_idx >> obs.u >> obs.v;
        observations.push_back(obs);
    }

    vector<Eigen::VectorXd> camera_params(num_cams);
    for (int i = 0; i < num_cams; ++i) {
        Eigen::VectorXd pose(6);
        for (int j = 0; j < 6; ++j)
            infile >> pose[j];
        camera_params[i] = pose;
    }

    // Map to store point ID to index mapping and reverse mapping
    unordered_map<int, int> point_id_to_index;
    unordered_map<int, int> index_to_point_id;
    vector<Eigen::Vector3d> points(num_points);
    for (int i = 0; i < num_points; ++i) {
        int pid;
        infile >> pid;
        infile >> points[i][0] >> points[i][1] >> points[i][2];
        point_id_to_index[pid] = i;
        index_to_point_id[i] = pid;
    }

    infile.close();

    // Setup optimizer
    typedef BlockSolver<BlockSolverTraits<6, 3>> Block;
    auto linearSolver = make_unique<LinearSolverDense<Block::PoseMatrixType>>();
    auto solver = new OptimizationAlgorithmLevenberg(make_unique<Block>(std::move(linearSolver)));

    SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // Camera intrinsics (example: fx=180, fy=180, cx=320, cy=240)
    CameraParameters* cam_params = new CameraParameters(180.0, Eigen::Vector2d(320, 240), 0);
    cam_params->setId(0);
    optimizer.addParameter(cam_params);

    // Add camera vertices
    for (int i = 0; i < num_cams; ++i) {
        VertexSE3Expmap* v = new VertexSE3Expmap();
        Eigen::Vector3d trans = camera_params[i].segment<3>(3);
        Eigen::Vector3d rot = camera_params[i].segment<3>(0);
        v->setId(i);
        v->setEstimate(SE3Quat(Eigen::AngleAxisd(rot.norm(), rot.normalized()).toRotationMatrix(), trans));
        if (i == 0) v->setFixed(true);  // Fix the first camera
        optimizer.addVertex(v);
    }

    // Add point vertices
    int point_id_offset = num_cams;
    for (int i = 0; i < num_points; ++i) {
        VertexPointXYZ* v = new VertexPointXYZ();
        v->setId(point_id_offset + i);
        v->setEstimate(points[i]);
        v->setMarginalized(true);
        optimizer.addVertex(v);
    }

    // Add edges
    for (const Obs& obs : observations) {
        EdgeProjectXYZ2UV* e = new EdgeProjectXYZ2UV();
        e->setVertex(0, optimizer.vertex(point_id_offset + point_id_to_index[obs.pt_idx]));
        e->setVertex(1, optimizer.vertex(obs.cam_idx));
        e->setMeasurement(Eigen::Vector2d(obs.u, obs.v));
        e->setInformation(Eigen::Matrix2d::Identity());
        e->setParameterId(0, 0);
        optimizer.addEdge(e);
    }

    cout << "Starting optimization with " << optimizer.vertices().size() << " vertices and "
         << optimizer.edges().size() << " edges." << endl;

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    // Output optimized camera poses
    // cout << "\nOptimized Camera Poses:\n";
    ofstream pose_file("../output/optimized_camera_poses.txt");
    for (int i = 0; i < num_cams; ++i) {
        VertexSE3Expmap* v = static_cast<VertexSE3Expmap*>(optimizer.vertex(i));
        SE3Quat pose = v->estimate();
        Eigen::Vector3d trans = pose.translation();
        Eigen::Vector3d rot = pose.rotation().toRotationMatrix().eulerAngles(0, 1, 2);
        // cout << "Camera " << i << ":\n";
        // cout << "  Translation: " << trans.transpose() << endl;
        // cout << "  Rotation (Euler angles): " << rot.transpose() << endl;
        pose_file << trans[0] << " " << trans[1] << " " << trans[2] << " "
                 << rot[0] << " " << rot[1] << " " << rot[2] << endl;
    }
    pose_file.close();

    // Output optimized robot poses
    // cout << "\nOptimized Robot Poses:\n";
    ofstream robot_pose_file("../output/optimized_robot_poses.txt");
    
    // Fixed camera-to-robot transform (T_robot_camera)
    // This describes how the camera is mounted on the robot:
    // - 0.2m forward along robot's x-axis
    // - 90 degree rotation around z-axis
    // - 90 degree rotation around x-axis
    Eigen::Matrix4d T_robot_camera;
    T_robot_camera <<   0,  0, 1, 0.2,
                       -1,  0, 0,   0,
                        0, -1, 0,   0,
                        0,  0, 0,   1;
    
    for (int i = 0; i < num_cams; ++i) {
        // Get optimized camera pose from bundle adjustment
        // This pose is in the camera's coordinate frame
        VertexSE3Expmap* v = static_cast<VertexSE3Expmap*>(optimizer.vertex(i));
        SE3Quat cam_pose = v->estimate();
        
        // Convert camera pose to 4x4 transformation matrix (T_world_camera)
        // This represents the camera's position and orientation in world frame
        Eigen::Matrix4d T_camera_camera = Eigen::Matrix4d::Identity();
        T_camera_camera.block<3,3>(0,0) = cam_pose.rotation().toRotationMatrix();
        T_camera_camera.block<3,1>(0,3) = cam_pose.translation();
        
        // Compute robot pose in world frame:
        // robot_pose = (T_robot_camera)^-1 * T_camera_camera
        // This transforms from:
        // World Frame <-- T_world_camera -- Camera Frame <-- (T_robot_camera)^-1 -- Robot Frame
        Eigen::Matrix4d T_world_robot = T_robot_camera.inverse() * T_camera_camera;
        
        // Extract translation and rotation from robot pose
        Eigen::Vector3d robot_trans = T_world_robot.block<3,1>(0,3);
        Eigen::Matrix3d robot_rot = T_world_robot.block<3,3>(0,0);
        
        // Extract theta (rotation around z-axis) using atan2
        double theta = atan2(robot_rot(1,0), robot_rot(0,0));
        
        // cout << "Robot " << i << ":\n";
        // cout << "  Translation: " << robot_trans.transpose() << endl;
        // cout << "  Theta (z-axis rotation): " << theta << endl;
        
        robot_pose_file << robot_trans[0] << " " << robot_trans[1] << " " << robot_trans[2] << " "
                       << theta << endl;
    }
    robot_pose_file.close();

    // Output optimized 3D points sorted by landmark ID
    // cout << "\nOptimized 3D Points:\n";
    ofstream points_file("../output/optimized_points.txt");
    map<int, Eigen::Vector3d> sorted_points;
    for (int i = 0; i < num_points; ++i) {
        VertexPointXYZ* v = static_cast<VertexPointXYZ*>(optimizer.vertex(point_id_offset + i));
        sorted_points[index_to_point_id[i]] = v->estimate();
    }
    
    for (const auto& [id, point] : sorted_points) {
        // cout << "Landmark " << id << ": " << point.transpose() << endl;
        points_file << id << " " << point[0] << " " << point[1] << " " << point[2] << endl;
    }
    points_file.close();

    return 0;
}

// TODO visualize results on view_path.py
// TODO compute error
// TODO can we force the z value of the robot or camera to be 0 or 0.2 respectively
// TODO incorporate Kalman filter?

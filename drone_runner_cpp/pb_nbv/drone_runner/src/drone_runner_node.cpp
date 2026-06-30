#include <ros/ros.h>
#include <pb_core/pbnbv.h>
#include <geometry_msgs/Pose.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <json/json.h>
#include <fstream>
#include <vector>

class DroneRunner {
private:
    pbnbv pb_nbv;
    std::string pcd_file_path;
    std::string output_dir;
    std::vector<Eigen::Matrix4d> path_history;
    std::vector<std::string> azimuths;
    std::vector<double> coverages;
    std::vector<int> voxel_gains;
    int step_count = 0;

public:
    DroneRunner() : pb_nbv() {
        char* work_dir_cstr = std::getenv("WORK_DIR");
        std::string work_dir = work_dir_cstr ? std::string(work_dir_cstr) : "/workspace/pb_nbv/";

        pcd_file_path = work_dir + "src/ground_first.pcd";
        output_dir = work_dir + "results/drone_runner/";

        system(("mkdir -p " + output_dir).c_str());
        ROS_INFO("DroneRunner initialized. PCD: %s", pcd_file_path.c_str());
    }

    bool run() {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path, *cloud) == -1) {
            ROS_ERROR("Failed to load PCD file: %s", pcd_file_path.c_str());
            return false;
        }
        ROS_INFO("Loaded PCD: %lu points", cloud->size());

        Eigen::Matrix4d current_pose = Eigen::Matrix4d::Identity();

        for (int i = 0; i < 16; i++) {
            step_count = i + 1;
            ROS_INFO("====== Step %d ======", step_count);

            Eigen::Matrix4d nbv;
            int is_terminal;

            if (i == 0) {
                is_terminal = pb_nbv.execute(nbv);
            } else {
                pb_nbv.capture(cloud, current_pose);
                is_terminal = pb_nbv.execute(nbv);
            }

            double azimuth = std::atan2(nbv(1,3), nbv(0,3)) * 180.0 / M_PI;
            if (azimuth < 0) azimuth += 360.0;
            double altitude = -nbv(2,3);
            double distance = std::sqrt(nbv(0,3)*nbv(0,3) + nbv(1,3)*nbv(1,3));

            ROS_INFO("Step %d: Az=%.1f° Alt=%.1f Dist=%.1f",
                     step_count, azimuth, altitude, distance);

            path_history.push_back(nbv);
            azimuths.push_back(std::to_string(static_cast<int>(azimuth*10)/10.0));
            coverages.push_back(0.0);
            voxel_gains.push_back(0);

            current_pose = nbv;

            if (is_terminal == 1) {
                ROS_INFO("Iteration terminated at step %d", step_count);
                break;
            }
        }

        saveResults();
        return true;
    }

private:
    void saveResults() {
        Json::Value root;
        root["num_steps"] = step_count;
        root["platform"] = "drone_with_pb_nbv";

        Json::Value path_array(Json::arrayValue);
        for (int i = 0; i < static_cast<int>(path_history.size()); i++) {
            Json::Value step;
            step["step"] = i + 1;
            step["x"] = path_history[i](0,3);
            step["y"] = path_history[i](1,3);
            step["z"] = path_history[i](2,3);
            step["azimuth"] = azimuths[i];
            path_array.append(step);
        }
        root["path"] = path_array;

        std::string output_file = output_dir + "result.json";
        std::ofstream ofs(output_file);
        ofs << root.toStyledString();
        ofs.close();

        ROS_INFO("Results saved to: %s", output_file.c_str());
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "drone_runner_node");
    google::InitGoogleLogging(argv[0]);

    char* work_dir_cstr = std::getenv("WORK_DIR");
    std::string work_dir = work_dir_cstr ? std::string(work_dir_cstr) : "/workspace/pb_nbv/";
    google::SetLogDestination(google::INFO, (work_dir + "src/pb_core/log/").c_str());
    FLAGS_stderrthreshold = google::INFO;

    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    DroneRunner runner;
    if (!runner.run()) {
        return 1;
    }

    spinner.stop();
    return 0;
}

#include <pb_core/pbnbv.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstdlib>

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
    int max_steps = 16;
    std::string run_id;

public:
    DroneRunner() : pb_nbv() {
        std::cout << "[DEBUG] DroneRunner constructor starting" << std::endl;
        fflush(stdout);

        char* work_dir_cstr = std::getenv("WORK_DIR");
        std::string work_dir = work_dir_cstr ? std::string(work_dir_cstr) : "/workspace/pb_nbv_minimal/";

        char* max_steps_cstr = std::getenv("MAX_STEPS");
        max_steps = max_steps_cstr ? std::stoi(max_steps_cstr) : 16;

        char* run_id_cstr = std::getenv("RUN_ID");
        run_id = run_id_cstr ? std::string(run_id_cstr) : "default";

        char* pcd_file_cstr = std::getenv("PCD_FILE");
        pcd_file_path = pcd_file_cstr ? std::string(pcd_file_cstr) : work_dir + "ground_first.pcd";
        output_dir = work_dir + "results/drone_runner/";

        system(("mkdir -p " + output_dir).c_str());
        std::cout << "[DroneRunner] M=" << max_steps << " RUN_ID=" << run_id << std::endl;
        fflush(stdout);
    }

    bool run() {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path, *cloud) == -1) {
            std::cerr << "[DroneRunner] Failed to load PCD: " << pcd_file_path << std::endl;
            return false;
        }
        std::cout << "[DroneRunner] Loaded " << cloud->size() << " points" << std::endl;

        Eigen::Matrix4d current_pose = Eigen::Matrix4d::Identity();

        for (int i = 0; i < max_steps; i++) {
            step_count = i + 1;
            std::cout << "====== Step " << step_count << " ======" << std::endl;

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

            std::cout << "  Az=" << azimuth << "° Alt=" << altitude << " Dist=" << distance << std::endl;

            path_history.push_back(nbv);
            azimuths.push_back(std::to_string(static_cast<int>(azimuth*10)/10.0));
            coverages.push_back(0.0);
            voxel_gains.push_back(0);

            current_pose = nbv;

            if (is_terminal == 1) {
                std::cout << "[DroneRunner] Terminated at step " << step_count << std::endl;
                break;
            }
        }

        saveResults();
        return true;
    }

private:
    void saveResults() {
        std::string output_file = output_dir + "result_" + run_id + ".txt";
        std::ofstream ofs(output_file);
        ofs << "num_steps: " << step_count << "\n";
        ofs << "platform: drone_with_pb_nbv\n\n";
        ofs << "Path:\n";
        for (int i = 0; i < static_cast<int>(path_history.size()); i++) {
            ofs << "Step " << (i+1) << ": "
                << "x=" << std::setprecision(3) << path_history[i](0,3) << " "
                << "y=" << path_history[i](1,3) << " "
                << "z=" << path_history[i](2,3) << " "
                << "azimuth=" << azimuths[i] << "\n";
        }
        ofs.close();
        std::cout << "[DroneRunner] Results saved to: " << output_file << std::endl;
    }
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    DroneRunner runner;
    bool ok = runner.run();
    // quick_exit skips static object destructors (avoids OctoMap 1.9.3 crash on exit)
    std::quick_exit(ok ? 0 : 1);
}

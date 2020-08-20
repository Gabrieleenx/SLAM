#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud.h"
#include "geometry_msgs/Point32.h"
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseStamped.h"

#include "sensor_msgs/ChannelFloat32.h"
#include <signal.h>

#include <thread>
#include <mutex>
#include <math.h>
#include <random>
#include <iostream> // for cout
#include <chrono>  // for timer
using namespace std;

// globals as not allowed to define in class :(
const static double map_resloution = 0.05; // in meters
const double pi = 2*acos(0.0);

// mutex for when copying data from local map to particles 
mutex mtx;

// map grid resolution 0.05 meters.
const static int map_grid_size_x = 10/0.05;// in meters
const static int map_grid_size_y = 10/0.05; // in meters
const static int map_grid_size_z = 4/0.05; // in met

// map grid resolution 0.05 meters.
//const static int particle_map_grid_size_x = 10/0.05;// in meters
//const static int particle_map_grid_size_y = 10/0.05; // in meters
//const static int particle_map_grid_size_z = 4/0.05; // in met
// Particles
struct map_particles {
    const static int p_map_grid_size_x = 15/0.05;// in meters
    const static int p_map_grid_size_y = 15/0.05; // in meters
    const static int p_map_grid_size_z = 4/0.05; // in met
    const static int map_legth_x = 15;
    const static int map_legth_y = 15;
    const static int map_legth_z = 4;
    const static int p_map_elements = p_map_grid_size_x*p_map_grid_size_y*p_map_grid_size_z;
    double pos_x = 7.5;
    double pos_y = 7.5;
    double pos_z = 1.5;
    double rot_z = 0.0;
    double weight = 1.0/50;
    uint8_t map[p_map_grid_size_x*p_map_grid_size_y*p_map_grid_size_z] =  {0};
};

// build local map 
double local_pos_xyz[3] = {(10/2), (10/2), (1.5)};
double local_euler_zyx[3] = { 0 };
double euler_zyx[3] = { 0 };
double pos_xyz[3] = {7.5, 7.5, 1.5}; // same as for particle system 
double pos_xyz_diff[3] = {0, 0, 0}; // same as for particle system 
double rot_z_diff = 0;
// global heap allocated memmory...
int map_elements = map_grid_size_x*map_grid_size_y*map_grid_size_z;
uint8_t* map_localization = new uint8_t[map_grid_size_x*map_grid_size_y*map_grid_size_z](); // ()initilizes to zero, hopefully. 
int8_t* map_local = new int8_t[map_grid_size_x*map_grid_size_y*map_grid_size_z](); // ()initilizes to zero, hopefully. 

int8_t* map_local_copy = new int8_t[map_grid_size_x*map_grid_size_y*map_grid_size_z](); // ()initilizes to zero, hopefully. 
int num_particles = 50;
map_particles* Particles = new map_particles[50];

void initilize_particles_map(){
    for (int i = 0; i < num_particles; i++){
        for  (int j = 0; j < Particles[0].p_map_elements; j++){
            Particles[i].map[j] = 50;
        }
    }
}

// converion x*map_grid_size_y*map_grid_size_z + y*map_grid_size_z + z

int index_conv(int x, int y, int z){
    return x*map_grid_size_y*map_grid_size_z + y*map_grid_size_z + z;
}
int index_conv_p(int x, int y, int z){
    return x*(15/0.05)*(4/0.05) + y*(4/0.05) + z;
}

void reverse_index_conv(int index_out[3], int index_in){
    int x;
    int y;
    int z;
    x = index_in/(map_grid_size_y*map_grid_size_z);
    y = (index_in-x*(map_grid_size_y*map_grid_size_z))/map_grid_size_z;
    z = (index_in- x*(map_grid_size_y*map_grid_size_z) -y*map_grid_size_z);
    index_out[0] = x;
    index_out[1] = y;
    index_out[2] = z;
}

void reverse_index_conv_p(int index_out[3], int index_in){
    int x;
    int y;
    int z;
    x = index_in/((15/0.05)*(4/0.05));
    y = (index_in-x*((15/0.05)*(4/0.05)))/(4/0.05);
    z = (index_in- x*((15/0.05)*(4/0.05)) -y*(4/0.05));
    index_out[0] = x;
    index_out[1] = y;
    index_out[2] = z;
}

class Slam{

public:
    // stores distance in mm, y = depth, x = horizontal, z = vertical
    const static int res_h = 640/4;
    const static int res_v = 480/4;
    int depth_z[res_v][res_h];
    int depth_y[res_v][res_h];
    int depth_x[res_v][res_h];

    double horizontal_distance[res_h];
    double vertical_distance[res_v];
    double focal_point = 385.0/4.0;

    double cloudpoints[res_h*res_v][3]; // x,y,z
    int cloudpoints_index = 0;

    string frame_id = "camera_depth_optical_frame";  

    int publish_cloud = 0; // nothing to publish yet

    const static int map_size_x = 10;// in meters
    const static int map_size_y = 10; // in meters
    const static int map_size_z = 4; // in meters


    chrono::high_resolution_clock::time_point time_last = chrono::high_resolution_clock::now();
    double euler_zyx_diff = 0;
    double euler_zyx_diff_gyr = 0;
    double euler_zyx_diff_gyr_old = 0;
    double euler_zyx_diff_scan[3] = { 0 };
    double pos_xyz_diff_scan[3] = {0};

    double img_time_stamp; 

    geometry_msgs::Vector3Stamped gyr_acc_buffer[50];
    int index_filter_buffer = 0;

    double gyr_acc_filter_zyx[3] = {};
    int no_new_gyr = 1;

    ros::Publisher pose_pub;

    Slam(ros::NodeHandle *n){ 
        // constructor 
        for(int i = 0; i < res_h; i++){
            horizontal_distance[i] = ((i+1)-(res_h/2))/focal_point;
        }
        for(int i = 0; i < res_v; i++){
            vertical_distance[i] = -1*((i+1)-(res_v/2))/focal_point;
        }

        pose_pub = n->advertise<geometry_msgs::PoseStamped>("/Pose", 1);

    }
    
    void callback(const sensor_msgs::Image::ConstPtr& msg){
        //chrono::high_resolution_clock::time_point time_last = chrono::high_resolution_clock::now();

        frame_id =  msg->header.frame_id; // should not change it but just in case

        resize_depth_to_array(msg->data);

        img_to_cloud();
        
        // syncronaize the data
        if(no_new_gyr == 0){
            img_time_stamp = (double)(msg->header.stamp.sec)+(double)(msg->header.stamp.nsec)*1e-9;
            sync_filter(img_time_stamp);
        }
        else
        {
            gyr_acc_filter_zyx[0] = euler_zyx[0] + euler_zyx_diff_scan[0]; 
        }

        scan_localization();

        scan_localization_map_update();

        build_local_map();

        pub_pose_func();

        // check computation time
        chrono::high_resolution_clock::time_point time_now = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(time_now - time_last);
        time_last = time_now;
        int time = duration.count();
        double fps = 1.0/time *1000000;
        //cout << "Fps "<< fps << " rot " << euler_zyx[0] << " x "<< pos_xyz[0] <<" y "<< pos_xyz[1]<<" z "<< pos_xyz[2] << "\n";


        publish_cloud = 1; 
    }

    void pub_pose_func(){
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.header.frame_id = frame_id;
        pose_msg.header.stamp = ros::Time::now();
        geometry_msgs::Pose pose_o;
        pose_o.orientation.w = cos((-euler_zyx[0]-pi/2)/2);
        pose_o.orientation.x =0;
        pose_o.orientation.y =sin((-euler_zyx[0]-pi/2)/2);
        pose_o.orientation.z = 0;
        pose_o.position.x = pos_xyz[0];
        pose_o.position.y = -pos_xyz[2];
        pose_o.position.z = pos_xyz[1];
        pose_msg.pose = pose_o;
        pose_pub.publish(pose_msg);

    }

    void filter_callback(const geometry_msgs::Vector3Stamped::ConstPtr& msg){
        //double time_ = (double)(gyr_acc_buffer[0].header.stamp.sec)+(double)(gyr_acc_buffer[0].header.stamp.nsec)*1e-9;
        //cout << "woring filter sub " <<msg->header.stamp << "  " << time_ << "\n"; // it does work...
        gyr_acc_buffer[index_filter_buffer] = *msg;
        index_filter_buffer += (index_filter_buffer < 49) ? 1 : -49;
        // initial before the buffer has been filled. 
        if (index_filter_buffer == 49){ 
            no_new_gyr = 0;
        }
    }
  
    void sync_filter(double img_time){
        double time_;
        double diff_time;
        double max_diff_time = 0.1;  
        for(int i = 0; i < 50; i ++){
            time_ = (double)(gyr_acc_buffer[i].header.stamp.sec)+(double)(gyr_acc_buffer[i].header.stamp.nsec)*1e-9;
            diff_time = abs(time_ - img_time);
            if (diff_time < max_diff_time){
                //cout << "diff_time " << diff_time << "\n"; 
                gyr_acc_filter_zyx[0] = gyr_acc_buffer[i].vector.z;
                gyr_acc_filter_zyx[1] = gyr_acc_buffer[i].vector.y;
                gyr_acc_filter_zyx[2] = gyr_acc_buffer[i].vector.x;
                max_diff_time = diff_time;
                no_new_gyr = 0;
            }
        }
        if(max_diff_time > 0.099){
            no_new_gyr = 1;
        }

    }

private:
    // scan localization 
    // map xyz
    //int map_localization[map_grid_size_x][map_grid_size_y][map_grid_size_z] = { 0 };
    const static int num_scan_points = 500;
    int sum_prob = 0;
    int sum_ = 0;
    int rand_index[num_scan_points];
    double rot_z_loc = 0;
    double rot_z;
    double rot_y = 0; // will be changeds to be from gyr acc fitler 
    double rot_x = 0.1; // will be changeds to be from gyr acc fitler 
    double Tx_loc = map_size_x/2;
    double Ty_loc = map_size_y/2;
    double Tz_loc = map_size_z/2;
    double Tx;
    double Ty;
    double Tz;
    double length = 0.25;
    int rotation_view = 15;
    int rot_resolution = 3;
    int r_steps = rotation_view/rot_resolution;
    int n_steps = length/map_resloution;

    double euler_zyx_scan[3] = { 0 };
    double pos_xyz_scan[3] = {map_size_x/2, map_size_y/2, map_size_z/2};

    double map_conv = 1/map_resloution;

    const static int num_of_old_points = 5;
    int points_scan_map[res_h*res_v*num_of_old_points]; // already multiplyed and indexed 
    int points_scan_map_index[num_of_old_points];
    int points_scan_map_life[num_of_old_points] = { };
    int random_index;

    int indx[3];
    double cloud_test[num_scan_points][3];
    double cloud_test_rot[num_scan_points][3];
    double cloud_test_loc[num_scan_points][3];
    int first_run = 1;
    int index_after_cov;
    int index_after_cov_2;

    void scan_localization(){
        rot_y = gyr_acc_filter_zyx[1];
        rot_x = gyr_acc_filter_zyx[2];
    
        // upadate loc map
        for(int i = 0; i < num_of_old_points; i++){
            if (points_scan_map_life[i] > 0){
                //cout << "adding to map " << "\n";
                for(int j = i*res_h*res_v; j < i*res_h*res_v + points_scan_map_index[i]; j++){
                    map_localization[points_scan_map[j]] = 1;
                }
            }
        }

        // generate 500 random index...
        random_device rd; // create random random seed
        mt19937 gen(rd()); // put the seed inte the random generator 
        uniform_int_distribution<> distr(0, cloudpoints_index); // create a distrobution
        for(int i = 0; i < num_scan_points; i++){
            random_index = distr(gen);
            cloud_test[i][0] = cloudpoints[random_index][0];
            cloud_test[i][1] = cloudpoints[random_index][1];
            cloud_test[i][2] = cloudpoints[random_index][2];
        }

        // localization
        int sum_prob = 0;

        for(int r = 0; r < r_steps; r++){
            rot_z = euler_zyx_scan[0] + (-(rotation_view-rot_resolution)/2 + 3*r)*pi/180;
            
            // rotate cloud
            rotate_point(cloud_test_rot, cloud_test, num_scan_points, rot_z, rot_y, rot_x);

            for(int x = 0; x < n_steps; x++){
                Tx = pos_xyz_scan[0] + x*map_resloution - (length-map_resloution)/2;
                for(int i = 0; i < num_scan_points; i++){
                    cloud_test_loc[i][0] = cloud_test_rot[i][0] + Tx;
                }

                for(int y = 0; y < n_steps; y++){
                    Ty = pos_xyz_scan[1] + y*map_resloution - (length-map_resloution)/2;
                    for(int i = 0; i < num_scan_points; i++){
                        cloud_test_loc[i][1] = cloud_test_rot[i][1] + Ty;
                    }

                    // will not be needed for the robot... as z is constant 
                    for(int z = 0; z < n_steps; z++){
                        Tz = pos_xyz_scan[2] + z*map_resloution - (length-map_resloution)/2;
                        sum_ = 0;
                        for(int i = 0; i < num_scan_points; i++){
                            cloud_test_loc[i][2] = cloud_test_rot[i][2] + Tz;
                             
                            indx[0] = round(cloud_test_loc[i][0]*map_conv);
                            indx[1] = round(cloud_test_loc[i][1]*map_conv);
                            indx[2] = round(cloud_test_loc[i][2]*map_conv);

                            index_after_cov_2 = index_conv(indx[0], indx[1], indx[2]);
                            if (index_after_cov_2 < 0 || index_after_cov_2 >= map_elements){
                                continue;
                            }
                            sum_ += map_localization[index_after_cov_2];
                            //cout << "all map are zero" << "\n";
                        }

                        if (sum_ > sum_prob){
                            //cout << "sum_ if " << sum_ <<"\n";
                            sum_prob = sum_;
                            rot_z_loc = rot_z;
                            Tx_loc = Tx;
                            Ty_loc = Ty;
                            Tz_loc = Tz;
                        }
                    }
                }
            }
        }

        
        pos_xyz_diff_scan[0] = cos(-euler_zyx_scan[0])*(Tx_loc - pos_xyz_scan[0]) - sin(-euler_zyx_scan[0])*(Ty_loc - pos_xyz_scan[1]);
        pos_xyz_diff_scan[1] = sin(-euler_zyx_scan[0])*(Tx_loc - pos_xyz_scan[0]) + cos(-euler_zyx_scan[0])*(Ty_loc - pos_xyz_scan[1]);
        pos_xyz_diff_scan[2] = Tz_loc - pos_xyz_scan[2];
        euler_zyx_diff_scan[0] = rot_z_loc - euler_zyx_scan[0];
        euler_zyx_scan[0] = rot_z_loc;
        pos_xyz_scan[0] = Tx_loc;
        pos_xyz_scan[1] = Ty_loc;
        pos_xyz_scan[2] = Tz_loc;

        // correction back to gyr acc filter
        
       


        if (abs(euler_zyx_diff_gyr_old-gyr_acc_filter_zyx[0]) < pi){
            euler_zyx_diff_gyr =  gyr_acc_filter_zyx[0] - euler_zyx_diff_gyr_old;
        }
        else if (gyr_acc_filter_zyx[0] < 0) {
            euler_zyx_diff_gyr =  2*pi+gyr_acc_filter_zyx[0] - euler_zyx_diff_gyr_old;
        }
        else
        {
            euler_zyx_diff_gyr =  -2*pi + gyr_acc_filter_zyx[0] - euler_zyx_diff_gyr_old;
        }   
        euler_zyx_diff_gyr_old = gyr_acc_filter_zyx[0];
        euler_zyx_diff = (euler_zyx_diff_gyr+ euler_zyx_diff_scan[0])/2;

        //geometry_msgs::Vector3 msg_c;
        //msg_c.z = euler_zyx[0] - gyr_acc_filter_zyx[0]; 

        //publish_gyr_correction.publish(msg_c);
    }

    //
    double cloudpoints_rot_update[res_h*res_v][3];

    double euler_zyx_update[3] = { 0 };
    double pos_xyz_update[3] = {map_size_x/2, map_size_y/2, map_size_z/2};
    double update_value = 0;

    void scan_localization_map_update(){
        // this should run after sensor fusion
        // delete old points in loc map

        update_value =  abs(pos_xyz_scan[0] - pos_xyz_update[0]) + abs(pos_xyz_scan[1] - pos_xyz_update[1])
                    +abs(pos_xyz_scan[2] - pos_xyz_update[2]) + 0.5*abs(euler_zyx_scan[0]-euler_zyx_update[0])
                    + 0.5*abs(rot_y - euler_zyx_update[1]) + 0.5*abs(rot_x - euler_zyx_update[2]);


        if (update_value > 0.15 || first_run == 1){

            
            for(int i = 0; i < num_of_old_points; i++){
                if (points_scan_map_life[i] == 1 || points_scan_map_life[i] <= 0){
                    if (points_scan_map_life[i] != 0){
                        for(int j = i*res_h*res_v; j < i*res_h*res_v + points_scan_map_index[i]; j++){
                            map_localization[points_scan_map[j]] = 0;
                        }
                    }
                    
                    rotate_point(cloudpoints_rot_update, cloudpoints, cloudpoints_index, euler_zyx_scan[0], rot_y, rot_x);
                    Translate_point(cloudpoints_rot_update, cloudpoints_rot_update, cloudpoints_index, pos_xyz_scan[0], pos_xyz_scan[1], pos_xyz_scan[2]);
                    // caclulate all index for new points.

                    // add index to list at place and update points_scan_map_index
                    points_scan_map_index[i] = 0;
                    for(int j= 0; j < cloudpoints_index; j++){
                        indx[0] = round((cloudpoints_rot_update[j][0]) * map_conv);
                        indx[1] = round((cloudpoints_rot_update[j][1]) * map_conv);
                        indx[2] = round((cloudpoints_rot_update[j][2]) * map_conv);
                        index_after_cov = index_conv(indx[0], indx[1], indx[2]);
                        if (index_after_cov < 0 || index_after_cov >= map_elements){
                            //cout << "index out of bound " << index_after_cov << "\n";
                        }
                        else{
                            points_scan_map[res_v*res_h*i + j] = index_after_cov;
                            points_scan_map_index[i] += 1;
                            //map_localization[index_after_cov] = 1;
                        }
                    
                    }
                    //cout << "num of index in map " << points_scan_map_index[i] << "\n";

                    // add 6 to life
                    points_scan_map_life[i] = 6;
                    //cout << "adding life to " << i << "\n"; 
                    // remove 1 life 
                    break;

                }
            }    
            for(int i = 0; i < num_of_old_points; i++){
                points_scan_map_life[i] += -1;
            }
            first_run = 0;

            pos_xyz_update[0] = pos_xyz_scan[0];
            pos_xyz_update[1] = pos_xyz_scan[1];
            pos_xyz_update[2] = pos_xyz_scan[2];
            euler_zyx_update[0] = euler_zyx_scan[0];
            euler_zyx_update[1] = rot_y;
            euler_zyx_update[2] = rot_x;

            cout << "update" << "\n"; 
        }

        if (abs(pos_xyz_scan[0] - (map_size_x/2)) > 2 || abs(pos_xyz_scan[1] -
             (map_size_y/2)) > 2|| abs(pos_xyz_scan[2] - (map_size_z/2)) > 1 ){

            cout << "reset localization " << endl;
            int offset_x = round(((map_size_x/2) - pos_xyz_scan[0])*map_conv);
            int offset_y = round(((map_size_y/2) - pos_xyz_scan[1])*map_conv);
            int offset_z = round(((map_size_z/2) - pos_xyz_scan[2])*map_conv);
            int new_index = 0;
            int offset_num = 0;
            int correction_offset = index_conv(offset_x, offset_y, offset_z);
            for(int i = 0; i < num_of_old_points; i++){

                // also zero out previous map...
                for(int j = i*res_h*res_v; j < i*res_h*res_v + points_scan_map_index[i]; j++){
                    map_localization[points_scan_map[j]] = 0;
                }
                offset_num = 0;
                for(int j = i*res_h*res_v; j < i*res_h*res_v + points_scan_map_index[i]; j++){
                    // ofest all index 
                    new_index = points_scan_map[j] + correction_offset;
                    if (new_index < 0 || new_index >= map_elements){
                        //cout << "index out of bound " << index_after_cov << "\n";
                        offset_num += 1;
                    }
                    else{
                        points_scan_map[j-offset_num] = new_index;
                        
                        //map_localization[index_after_cov] = 1;
                    }

                }
                points_scan_map_index[i] += -offset_num;
            }
            
            pos_xyz_update[0] += (map_size_x/2) - pos_xyz_scan[0];
            pos_xyz_scan[0] =  (map_size_x/2);
            pos_xyz_update[1] += (map_size_y/2) - pos_xyz_scan[1];
            pos_xyz_scan[1] =  (map_size_y/2);
            pos_xyz_update[2] += (map_size_z/2) - pos_xyz_scan[2];
            pos_xyz_scan[2] =  (map_size_z/2);


        }
    
    }
    
    //
    double Rot_M[3][3];
    // rotates points 
    void rotate_point(double new_point[][3], double point[][3], int num_points, double z, double y, double x){
        // need to be implemented...
        Rot_M[0][0] = cos(y)*cos(z);
        Rot_M[0][1] = cos(z)*sin(x)*sin(y) - cos(x)*sin(z);
        Rot_M[0][2] = sin(x)*sin(z) + cos(x)*cos(z)*sin(y);
        Rot_M[1][0] = cos(y)*sin(z);
        Rot_M[1][1] = cos(x)*cos(z) + sin(x)*sin(y)*sin(z);
        Rot_M[1][2] = cos(x)*sin(y)*sin(z) - cos(z)*sin(x);
        Rot_M[2][0] = -sin(y);
        Rot_M[2][1] = cos(y)*sin(x);
        Rot_M[2][2] = cos(x)*cos(y);

        for(int i = 0; i < num_points; i++){
            new_point[i][0] = Rot_M[0][0] * point[i][0] + Rot_M[0][1] * point[i][1] + Rot_M[0][2] * point[i][2];
            new_point[i][1] = Rot_M[1][0] * point[i][0] + Rot_M[1][1] * point[i][1] + Rot_M[1][2] * point[i][2];
            new_point[i][2] = Rot_M[2][0] * point[i][0] + Rot_M[2][1] * point[i][1] + Rot_M[2][2] * point[i][2];
        }
    }

    void Translate_point(double new_point[][3], double point[][3], int num_points, double x, double y, double z){
        for(int i = 0; i < num_points; i++){
            new_point[i][0] = point[i][0] + x;
            new_point[i][1] = point[i][1] + y;
            new_point[i][2] = point[i][2] + z;
        }
    }

    // resize depth image uint8_t to array int by mean over 4x4 and igoring zeors
    void resize_depth_to_array(const vector<uint8_t> &image){
        for(int i = 0; i < res_v; i++){
            for(int j = 0; j < res_h; j++){
                depth_y[i][j] = calc_mean_16(image, i, j);
                // matix multply to calculate depth_x and depth_z
                depth_x[i][j] = depth_y[i][j] * horizontal_distance[j];
                depth_z[i][j] = depth_y[i][j] * vertical_distance[i];
            }
        }
    }

    // variables for calc_mean_16
    int int8_to_int;
    int int_sum;
    int int8_index;
    int int8_idx_l;
    int devide_sum;
    // calulates the mean of a 4x4 square from a image and ignores 0 zeros.
    int calc_mean_16(const vector<uint8_t> &image, int z, int x){
        int_sum = 0;
        devide_sum  = 0;
        int8_index = z*4*1280+x*8; // idex in one dimetional vector
        for(int i = 0; i < 4;  i++){
            int8_idx_l = int8_index + i*1280; // 640*2       
            for(int j = 0; j < 4; j++) {
                int8_to_int = (image[int8_idx_l+1] << 8 )| image[int8_idx_l]; // int16 to int, 
                int8_idx_l += 2; // two int8 to build int16 
                int_sum += int8_to_int; // sum all 16 values
                devide_sum += (int8_to_int != 0) ? 1 : 0; // how many values that are not 0
            }
        }
        return (devide_sum != 0) ? int_sum/devide_sum : 0; // returns mean
    }
     
    // calculate cloudpoints
    
    void img_to_cloud(){
        cloudpoints_index = 0; 
        for(int i = 0; i < res_v; i++){
            for(int j = 0; j < res_h; j++){
                if (depth_y[i][j] != 0){
                    cloudpoints[cloudpoints_index][0] = depth_x[i][j] / 1000.0;
                    cloudpoints[cloudpoints_index][1] = depth_y[i][j] / 1000.0;
                    cloudpoints[cloudpoints_index][2] = depth_z[i][j] / 1000.0;
                    cloudpoints_index += 1;
                }

            }
        }

    }

    // calc norm between two points. 
    double norm(double x1, double x2, double y1, double y2, double z1, double z2){
        return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
    }



    void build_local_map(){
        mtx.lock();
        local_pos_xyz[0] += pos_xyz_diff_scan[0];
        local_pos_xyz[1] += pos_xyz_diff_scan[1];
        local_pos_xyz[2] += pos_xyz_diff_scan[2];
        local_euler_zyx[0] += euler_zyx_diff;
        // roatate points
        rotate_point(cloudpoints_rot_update, cloudpoints, cloudpoints_index, local_euler_zyx[0], rot_y, rot_x);              
        // translate points
        Translate_point(cloudpoints_rot_update, cloudpoints_rot_update, cloudpoints_index, 
                local_pos_xyz[0], local_pos_xyz[1], local_pos_xyz[2]);
        for(int i = 0; i < cloudpoints_index; i++){
            // check distance, i.e reject if more then 3 meters away. 
            double norm_ = norm(cloudpoints_rot_update[i][0], local_pos_xyz[0], cloudpoints_rot_update[i][1],
                                 local_pos_xyz[1], cloudpoints_rot_update[i][2], local_pos_xyz[2]);
            
            if(norm_ > 3){
                continue;
            }
            // point to index
            indx[0] = round((cloudpoints_rot_update[i][0]) * map_conv);
            indx[1] = round((cloudpoints_rot_update[i][1]) * map_conv);
            indx[2] = round((cloudpoints_rot_update[i][2]) * map_conv);
            index_after_cov = index_conv(indx[0], indx[1], indx[2]);
            if (index_after_cov < 0 || index_after_cov >= map_elements){
                //cout << "index out of bound " << index_after_cov << "\n";
            }
            else{
                // index to map
                map_local[index_after_cov] += 10;

                // clip map. 
                if(map_local[index_after_cov] > 50){
                    map_local[index_after_cov] = 50;
                }
                //else if (map_local[index_after_cov] < -50){
                //    map_local[index_after_cov] = -50;
                //}

            }
        }
        //ray cast
        
        random_device rd_l; // create random random seed
        mt19937 gen_l(rd_l()); // put the seed inte the random generator 
        uniform_int_distribution<> distr_l(0, cloudpoints_index); // create a distrobution
        double ray_cast_point[3];
        for(int i = 0; i<2400; i++){
            random_index = distr_l(gen_l);
            ray_cast_point[0] = cloudpoints_rot_update[random_index][0];
            ray_cast_point[1] = cloudpoints_rot_update[random_index][1];
            ray_cast_point[2] = cloudpoints_rot_update[random_index][2];
            double norm_ = norm(ray_cast_point[0], local_pos_xyz[0], ray_cast_point[1],
                                 local_pos_xyz[1], ray_cast_point[2], local_pos_xyz[2]);
            //double length = norm_*map_conv;
            int ray_iteration = round((norm_*map_conv));
            if (norm_ > 3){
                int ray_iteration = 3*map_conv;
            }
 
            double ray_dx = (ray_cast_point[0] - local_pos_xyz[0])/(norm_*map_conv);
            double ray_dy = (ray_cast_point[1] - local_pos_xyz[1])/(norm_*map_conv);
            double ray_dz = (ray_cast_point[2] - local_pos_xyz[2])/(norm_*map_conv);

            for (int j = 1; j < ray_iteration; j++){

                indx[0] = round(( local_pos_xyz[0] + ray_dx*j) * map_conv);
                indx[1] = round((local_pos_xyz[1] + ray_dy*j) * map_conv);
                indx[2] = round((local_pos_xyz[2] + ray_dz*j) * map_conv);
                index_after_cov = index_conv(indx[0], indx[1], indx[2]);
                if (index_after_cov < 0 || index_after_cov >= map_elements){
                }
                else{
                    // index to map
                    map_local[index_after_cov] += -3;

                    // clip map. 
                    if(map_local[index_after_cov] < -50){
                        map_local[index_after_cov] = -50;
                    }
                }
            } 

        }

        update_realtime_pos();

        mtx.unlock();
        
    }

    void update_realtime_pos(){
        pos_xyz[0] += cos(euler_zyx[0])*pos_xyz_diff_scan[0] - sin(euler_zyx[0])*pos_xyz_diff_scan[1] + pos_xyz_diff[0];
        pos_xyz[1] += sin(euler_zyx[0])*pos_xyz_diff_scan[0] + cos(euler_zyx[0])*pos_xyz_diff_scan[1] + pos_xyz_diff[1];
        pos_xyz[2] += pos_xyz_diff_scan[2] + pos_xyz_diff[2];
        euler_zyx[0] += euler_zyx_diff + rot_z_diff;
        if (euler_zyx[0] > pi){
            euler_zyx[0] += -2*pi;
        }
        else if(euler_zyx[0] < -pi){
            euler_zyx[0] += 2*pi;
        }
        rot_z_diff = 0; 
        pos_xyz_diff[0] = 0;
        pos_xyz_diff[1] = 0;
        pos_xyz_diff[2] = 0;

    }
};

void rot_matrix(double matrix[3][3], double z, double y, double x){
    // need to be implemented...
    matrix[0][0] = cos(y)*cos(z);
    matrix[0][1] = cos(z)*sin(x)*sin(y) - cos(x)*sin(z);
    matrix[0][2] = sin(x)*sin(z) + cos(x)*cos(z)*sin(y);
    matrix[1][0] = cos(y)*sin(z);
    matrix[1][1] = cos(x)*cos(z) + sin(x)*sin(y)*sin(z);
    matrix[1][2] = cos(x)*sin(y)*sin(z) - cos(z)*sin(x);
    matrix[2][0] = -sin(y);
    matrix[2][1] = cos(y)*sin(x);
    matrix[2][2] = cos(x)*cos(y);
}

void rotate(double new_point[3], int point[3], double Rot_M[3][3]){
    new_point[0] = Rot_M[0][0] * point[0] + Rot_M[0][1] * point[1] + Rot_M[0][2] * point[2];
    new_point[1] = Rot_M[1][0] * point[0] + Rot_M[1][1] * point[1] + Rot_M[1][2] * point[2];
    new_point[2] = Rot_M[2][0] * point[0] + Rot_M[2][1] * point[1] + Rot_M[2][2] * point[2];
}

void reset_local_map(){
    for(int i = 0; i < map_elements; i++){
        map_local[i] = 0;
    }   
}

int index_high_weight = 0;
void resample(){
    double last_high_weight = 0;

    double resample_list[num_particles+1];
    int resample_index[num_particles];
    double sum_ = 0;
    resample_list[0] = 0;
    for(int i = 0; i < num_particles; i++){
        sum_ += Particles[i].weight;
        resample_list[i+1] = sum_;
    }
    random_device rd; // create random random seed
    mt19937 gen(rd()); // put the seed inte the random generator 
    uniform_real_distribution<double> distr(0.0, 1.0); // create a distrobution

    for(int i = 0; i < num_particles; i++){
        double rand_val = distr(gen);
        for (int j = 1; j < num_particles+1; j++){
            if(rand_val > resample_list[j-1] && rand_val < resample_list[j]){
                resample_index[i] = j-1;
                break;
            }
        }
    }
    // organize to minimize copying data.
    int resample_index_org[num_particles] = { 1 };
    int resample_index_taken[num_particles] = { 0 };
    for (int i = 0; i < num_particles; i++){
        resample_index_taken[resample_index[i]] = 1;
    }
    for (int i = 0; i < num_particles; i++){
        if(resample_index[i] == resample_index_org[resample_index[i]]){
            for(int j = 0; j < num_particles; j++){
                if (resample_index_taken[j] ==0){
                    resample_index_org[j] = resample_index[i];
                    resample_index_taken[j] = 1;
                }
            }
        }
        else
        {
            resample_index_org[resample_index[i]] = resample_index[i];
        }
        
    }

    // copy data
    for(int i = 0; i < num_particles; i++){
        if (resample_index_org[i] == i){
            continue;
        }
        else{
            Particles[i] = Particles[resample_index_org[i]];
        }
    }

        // save index with higest weight 
    for(int i = 0; i < num_particles; i++){
        if (Particles[i].weight > last_high_weight){
            index_high_weight = i;
            last_high_weight = Particles[i].weight;
        }
    }
}

int add_points = 1;
double pos_last_update[3] = {7.5, 7.5, 1.5};
double rot_last_update = 0;

void particle_map(){
    double p_rot_local_z;
    double p_pos_local_xyz[3];
    double p_rot_z;
    double p_pos_xyz[3];
    // copy data needed 
    mtx.lock();
    memcpy(map_local_copy, map_local, map_elements);
    p_rot_local_z = local_euler_zyx[0];
    memcpy(&p_pos_local_xyz, &local_pos_xyz, sizeof(local_pos_xyz));
    p_rot_z = euler_zyx[0];
    memcpy(&p_pos_xyz, &pos_xyz, sizeof(p_pos_xyz));
    // reset local map  7
    reset_local_map();
    local_euler_zyx[0] = 0;
    local_pos_xyz[0] = 5;
    local_pos_xyz[1] = 5;
    local_pos_xyz[2] = 1.5;
    // unclock mutex
    mtx.unlock();

    double Rot_matrix[3][3];
    double rot_indx[3];
    int local_indx[3];
    int P_map_indx;
    random_device rd; // create random random seed
    mt19937 gen(rd()); // put the seed inte the random generator 
    normal_distribution<float> pos_d(0, 0.03); // create a distrobution
    normal_distribution<float> rot_d(0, 0.02); // create a distrobution
    double sum_weights = 0;
    int num_match = 0;
    int sum_match[num_particles] = {0};
    for(int i = 0; i < num_particles; i++){
        rot_matrix(Rot_matrix, Particles[i].rot_z, 0.0, 0.0);
        // translate indx
        double T_x = Particles[i].pos_x * 1/map_resloution;
        double T_y = Particles[i].pos_y * 1/map_resloution;
        double T_z = Particles[i].pos_z * 1/map_resloution;
        num_match = 0;

        int new_value;
        for(int j = 0; j < map_elements; j++){
            // if 0 do nothing 
            if(map_local_copy[j] == 0){
                continue;
            }
            // rotate and translate
            reverse_index_conv(local_indx, j);
            local_indx[0] += -100; // 5/0.05
            local_indx[1] += -100; // 5/0.05
            local_indx[2] += -30; // 1.5/0.05
            rotate(rot_indx, local_indx, Rot_matrix);

            P_map_indx = index_conv_p(round(rot_indx[0] + T_x), round(rot_indx[1] + T_y), round(rot_indx[2] + T_z));
            // check if index is inside of map.
            if(P_map_indx < 0 || P_map_indx > Particles[i].p_map_elements){
                continue;
            }
            //if larger then 20; match 
            if(map_local_copy[j] > 20){
                sum_match[i] += Particles[i].map[P_map_indx];
                num_match += 1;
            }

            // add to map 
            if (add_points == 0){
                continue;
            }
            new_value = Particles[i].map[P_map_indx] + map_local_copy[j];
            if (new_value < 0){
                Particles[i].map[P_map_indx] = 0;
            }
            else if (new_value > 100){
                Particles[i].map[P_map_indx] = 100;
            }
            else{
                Particles[i].map[P_map_indx] = new_value;
            }
            //}
           

        }
       
        // update position and orientation
        Particles[i].pos_x += cos(Particles[i].rot_z)*(p_pos_local_xyz[0]-5) - sin(Particles[i].rot_z)*(p_pos_local_xyz[1]-5) + pos_d(gen); // add noise
        Particles[i].pos_y += sin(Particles[i].rot_z)*(p_pos_local_xyz[0]-5) + cos(Particles[i].rot_z)*(p_pos_local_xyz[1]-5) + pos_d(gen);
        Particles[i].pos_z += p_pos_local_xyz[2]-1.5 + rot_d(gen);
        Particles[i].rot_z += p_rot_local_z+ rot_d(gen);

    }


    // update weights
    double smalest_sum = 1000000000000;
    for(int i = 0; i < num_particles; i++){
        if (smalest_sum > sum_match[i]){
            smalest_sum = sum_match[i];
        }
    }

    for (int i = 0; i < num_particles; i++){
        Particles[i].weight = Particles[i].weight * (sum_match[i]-0.99*smalest_sum)*(sum_match[i]-0.99*smalest_sum)/num_match;
        sum_weights += Particles[i].weight;
    }

    // normalize weights

    cout << "weights" << " " << endl; 
    for(int i = 0; i < num_particles; i++){
        Particles[i].weight = (Particles[i].weight)/(sum_weights);
        cout << Particles[i].weight << " "; 
    }
    cout << "weights" << " " << endl; 
    // resample 
    double moved = abs(p_pos_xyz[0] - pos_last_update[0])+ abs(p_pos_xyz[1] - pos_last_update[1])+ 
        abs(p_pos_xyz[2] - pos_last_update[2])+  abs(p_rot_z - rot_last_update);
    if (moved > 0.4){
        add_points = 1;
        pos_last_update[0] = p_pos_xyz[0];
        pos_last_update[1] = p_pos_xyz[1];
        pos_last_update[2] = p_pos_xyz[2];
        rot_last_update = p_rot_z;
        //resample();
    }
    else {
        add_points = 0; 
    }
    resample();

    pos_xyz_diff[0] = Particles[index_high_weight].pos_x - p_pos_xyz[0];
    pos_xyz_diff[1] = Particles[index_high_weight].pos_y - p_pos_xyz[1];
    pos_xyz_diff[2] = Particles[index_high_weight].pos_z - p_pos_xyz[2];
    rot_z_diff = Particles[index_high_weight].rot_z - p_rot_z;
    cout << "after resample"<< Particles[index_high_weight].pos_x << " " << Particles[index_high_weight].pos_y << " " << Particles[index_high_weight].pos_z << " " <<Particles[index_high_weight].rot_z  
        << " index weight " <<  Particles[index_high_weight].weight << " " << index_high_weight << endl;
}

void publish_cloud_points(const Slam& slam, ros::Publisher& pub, uint32_t seq){
    

    // creates pointcloud message and publish it. 
    if(slam.publish_cloud == 1){
        //particle_map();
        sensor_msgs::PointCloud msg;
        msg.header.frame_id = slam.frame_id;
        msg.header.stamp = ros::Time::now();
        msg.header.seq = seq;

        geometry_msgs::Point32 point32;
        sensor_msgs::ChannelFloat32 point_rgb;
        uint8_t red;
        uint8_t green;
        uint8_t blue;

        point_rgb.name = "rgb";

        uint32_t packed_int_rgb;
        float packed_float_rgb;
        // for puublish input 
        /*
        for(int i = 0; i < slam.cloudpoints_index; i++){
            point32.x = slam.cloudpoints[i][0];
            point32.y = -slam.cloudpoints[i][2];
            point32.z = slam.cloudpoints[i][1];

            msg.points.push_back(point32);

            // color channels, not working 100%, makes bands instead of gradient..
            red = (slam.cloudpoints[i][2]+2)*40;
            green = 100;
            blue = 255 - ((slam.cloudpoints[i][2]+2)*40);
            // probably someting wrong in the packing
            packed_int_rgb = (red <<16) | (green << 8) | blue;
            packed_float_rgb = (float) ( ((double) packed_int_rgb)/((double) (1<<24)) );
            point_rgb.values.push_back(packed_float_rgb);
            
        }
        */ /*
        for(int i = 0; i < map_elements; i++){
            
            if (map_local[i] > 30){
                int index_map_pub[3];
                reverse_index_conv(index_map_pub, i);
                point32.x = index_map_pub[0]*map_resloution;
                point32.y = -index_map_pub[2]*map_resloution;
                point32.z = index_map_pub[1]*map_resloution;
                msg.points.push_back(point32);
                red = (uint8_t)((index_map_pub[2]*map_resloution)*60);
                green = (uint8_t)100;
                blue = (uint8_t)(255 - ((index_map_pub[2]*map_resloution)*60));

                packed_int_rgb = (red <<16) | (green << 8) | blue;
                packed_float_rgb = *reinterpret_cast<float*>(&packed_int_rgb);
                //packed_float_rgb = (float) ( ((double) packed_int_rgb)/((double) (1<<24)) );
                point_rgb.values.push_back(packed_float_rgb);
            }

            
        }*/


        cout << "publish map " << endl;
        
        // PUBLISH MAP
        for(int i = 0; i < Particles[0].p_map_elements; i+=4){
            
            if (Particles[index_high_weight].map[i] > 80){
                int index_map_pub[3];
                reverse_index_conv_p(index_map_pub, i);
                point32.x = index_map_pub[0]*map_resloution;
                point32.y = -index_map_pub[2]*map_resloution;
                point32.z = index_map_pub[1]*map_resloution;
                msg.points.push_back(point32);
                //red = (uint8_t)((index_map_pub[2]*map_resloution)*60);
                //green = (uint8_t)100;
                //blue = (uint8_t)(255 - ((index_map_pub[2]*map_resloution)*60));

                //packed_int_rgb = (red <<16) | (green << 8) | blue;
                //packed_float_rgb = *reinterpret_cast<float*>(&packed_int_rgb);
                //packed_float_rgb = (float) ( ((double) packed_int_rgb)/((double) (1<<24)) );
                //point_rgb.values.push_back(packed_float_rgb);
            }

            
        }
        

        msg.channels.push_back(point_rgb);
        pub.publish(msg);

    }

}

void mySigintHandler(int sig){
    cout << " Shutdown initilized " << "\n";
    ros::shutdown(); 
    // do pre shutdown, like delete heap memmory..
    delete[] map_localization; 
    delete[] map_local;
    delete[] Particles;
    delete[] map_local_copy;
    // does result in a segmentation fault as other 
    //thredes will look for this at the moment of shutdown. 

    //
    cout << " Shutdown done " << "\n";

    
}



int main(int argc, char **argv){


    ros::init(argc, argv, "Slam", ros::init_options::NoSigintHandler);

    ros::NodeHandle n;
    Slam slam = Slam(&n);
    initilize_particles_map(); //
    signal(SIGINT, mySigintHandler);

    ros::AsyncSpinner spinner(0);
    spinner.start();

    ros::Publisher publish_point_cloud = n.advertise<sensor_msgs::PointCloud>("/PointCloud", 1);

    ros::Subscriber sub = n.subscribe("/camera/depth/image_rect_raw", 2, &Slam::callback, &slam);
    ros::Subscriber filter_sub = n.subscribe("/gyro_acc_filter", 16, &Slam::filter_callback, &slam);

    ros::Rate loop_rate(0.5);

    uint32_t seq = 1;

    while (ros::ok()){
        publish_cloud_points(slam, publish_point_cloud, seq);

        ros::spinOnce();
        if(slam.publish_cloud == 1){
            particle_map();
        }
        loop_rate.sleep();
        ++seq;
    }
    ros::waitForShutdown();



    return 0;

}






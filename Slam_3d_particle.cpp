#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud.h"
#include "geometry_msgs/Point32.h"
#include "sensor_msgs/ChannelFloat32.h"

#include <random>
#include <iostream> // for cout
#include <chrono>  // for timer
using namespace std;

// globals as not allowed to define in class :(
const static double map_resloution = 0.05; // in meters
const double pi = 2*acos(0.0);

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

    const static int map_size_x = 30;// in meters
    const static int map_size_y = 30; // in meters
    const static int map_size_z = 6; // in meters

    // map grid resolution 0.05 meters.
    const static int map_grid_size_x = 30/0.05;// in meters
    const static int map_grid_size_y = 30/0.05; // in meters
    const static int map_grid_size_z = 6/0.05; // in meters

    double euler_zyx[3] = { 0 };
    double pos_xyz[3] = {map_size_x/2, map_size_y/2, map_size_z/2};

    double euler_zyx_diff_scan[3] = { 0 };
    double pos_xyz_diff_scan[3] = {0};

    Slam(){ 
        // constructor 
        for(int i = 0; i < res_h; i++){
            horizontal_distance[i] = ((i+1)-(res_h/2))/focal_point;
        }
        for(int i = 0; i < res_v; i++){
            vertical_distance[i] = -1*((i+1)-(res_v/2))/focal_point;
        }
    }
    
    
    void callback(const sensor_msgs::Image::ConstPtr& msg){
        chrono::high_resolution_clock::time_point time_last = chrono::high_resolution_clock::now();

        frame_id =  msg->header.frame_id; // should not change it but just in case

        resize_depth_to_array(msg->data);

        img_to_cloud();




        // will be included in paricles later
        euler_zyx[0] += euler_zyx_diff_scan[0];
        pos_xyz[0] += pos_xyz_diff_scan[0];
        pos_xyz[1] += pos_xyz_diff_scan[1];
        pos_xyz[2] += pos_xyz_diff_scan[2];

        // check computation time
        chrono::high_resolution_clock::time_point time_now = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(time_now - time_last);
        //time_last = time_now;
        int time = duration.count();
        double fps = 1.0/time *1000000;
        cout << "Fps "<< fps << " " << depth_y[31][22] << " "<< depth_x[31][22] <<" "<< depth_z[31][22] << "\n";


        publish_cloud = 1; // will be moved to somewhere else when the map has been implemented 
    }

private:

    int map_localization[map_grid_size_x][map_grid_size_y][map_grid_size_z] = { 0 };
    const static int num_scan_points = 500;
    int sum_prob = 0;
    int sum_ = 0;
    //int rand_index[num_scan_points];
    double rot_z_loc = 0;
    double rot_z;
    double rot_y = 0; // will be changeds to be from gyr acc fitler 
    double rot_x = 0; // will be changeds to be from gyr acc fitler 
    double Tx_loc = map_size_x/2;
    double Ty_loc = map_size_y/2;
    double Tz_loc = map_size_z/2;
    double Tx;
    double Ty;
    double Tz;
    double length = 0.25;
    int rotation_view = 45;
    int rot_resolution = 3;
    int r_steps = rotation_view/rot_resolution;
    int n_steps = length/map_resloution;

    double map_conv = 1/map_resloution;
    int point_loc_new[3];

    const static int num_of_old_points = 5;
    int points_scan_map[res_h*res_v*num_of_old_points][3]; // already multiplyed and indexed 
    int points_scan_map_index[num_of_old_points];
    int points_scan_map_life[num_of_old_points] = { 0 };
    int random_index;

    int indx[3];
    double cloud_test[num_scan_points][3];
    double cloud_test_rot[num_scan_points][3];
    double cloud_test_loc[num_scan_points][3];
    int firts_run = 1;

    void scan_localization(){

        if(firts_run == 0){

            // upadate loc map
       
            for(int i = 0; i < cloudpoints_index; i++){
                point_loc_new[0] = round(cloudpoints[i][0] * map_conv);
                point_loc_new[1] = round(cloudpoints[i][1] * map_conv);
                point_loc_new[2] = round(cloudpoints[i][2] * map_conv);
                map_localization[point_loc_new[0]][point_loc_new[1]][point_loc_new[2]] = 1;
            }


            // localization
            int sum_prob = 0;

            for(int r = 0; r < r_steps; r++){
                rot_z = (-rotation_view/2 + 3*r)*pi/180;
                // rotate cloud
                rotate_point(cloud_test_rot, cloud_test, num_scan_points, rot_z, rot_y, rot_x);

                for(int x = 0; x < n_steps; x++){
                    Tx = x*map_resloution - length/2;
                    for(int i = 0; i < num_scan_points; i++){
                        cloud_test_loc[i][0] = cloud_test_rot[i][0] + Tx;
                    }

                    for(int y = 0; y < n_steps; y++){
                        Ty = y*map_resloution - length/2;
                        for(int i = 0; i < num_scan_points; i++){
                            cloud_test_loc[i][1] = cloud_test_rot[i][1] + Ty;
                        }

                        // will not be needed for the robot... as z is constant 
                        for(int z = 0; z < n_steps; z++){
                            Tz = z*map_resloution - length/2;
                            sum_ = 0;
                            for(int i = 0; i < num_scan_points; i++){
                                cloud_test_loc[i][2] = cloud_test_rot[i][2] + Tz;
                                
                                indx[0] = round(cloud_test_loc[i][0]*map_conv);
                                indx[1] = round(cloud_test_loc[i][1]*map_conv);
                                indx[2] = round(cloud_test_loc[i][2]*map_conv);

                                if (indx[0] < 0 || indx[0] >= map_grid_size_x){
                                    continue;
                                }
                                if (indx[1] < 0 || indx[1] >= map_grid_size_y){
                                    continue;
                                }
                                if (indx[2] < 0 || indx[2] >= map_grid_size_z){
                                    continue;
                                }
                                sum_ += map_localization[indx[0]][indx[1]][indx[2]];

                            }

                            if (sum_ > sum_prob){
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

            euler_zyx_diff_scan[0] = -rot_z_loc;
            pos_xyz_diff_scan[0] = -Tx_loc;
            pos_xyz_diff_scan[1] = -Ty_loc;
            pos_xyz_diff_scan[2] = -Tz_loc;

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
        firts_run = 0;
    }

    /*
    // scan localization 
    // map xyz
    int map_localization[map_grid_size_x][map_grid_size_y][map_grid_size_z] = { 0 };
    const static int num_scan_points = 500;
    int sum_prob = 0;
    int sum_ = 0;
    int rand_index[num_scan_points];
    double rot_z_loc = 0;
    double rot_z;
    double rot_y = 0; // will be changeds to be from gyr acc fitler 
    double rot_x = 0; // will be changeds to be from gyr acc fitler 
    double Tx_loc = map_size_x/2;
    double Ty_loc = map_size_y/2;
    double Tz_loc = map_size_z/2;
    double Tx;
    double Ty;
    double Tz;
    double length = 0.25;
    int rotation_view = 45;
    int rot_resolution = 3;
    int r_steps = rotation_view/rot_resolution;
    int n_steps = length/map_resloution;

    double map_conv = 1/map_resloution;

    const static int num_of_old_points = 5;
    int points_scan_map[res_h*res_v*num_of_old_points][3]; // already multiplyed and indexed 
    int points_scan_map_index[num_of_old_points];
    int points_scan_map_life[num_of_old_points] = { 0 };
    int random_index;

    int indx[3];
    double cloud_test[num_scan_points][3];
    double cloud_test_rot[num_scan_points][3];
    double cloud_test_loc[num_scan_points][3];

    void scan_localization(){
        // upadate loc map
        for(int i = 0; i < num_of_old_points; i++){
            if (points_scan_map_life[i] != 0){
                for(int j = i*res_h*res_v; j < i*res_h*res_v + points_scan_map_index[i]; j++){
                    map_localization[points_scan_map[j][0]][points_scan_map[j][1]][points_scan_map[j][2]] = 1;
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
            rot_z = euler_zyx[0] + (-rotation_view/2 + 3*r)*pi/180;
            
            // rotate cloud
            rotate_point(cloud_test_rot, cloud_test, num_scan_points, rot_z, rot_y, rot_x);

            for(int x = 0; x < n_steps; x++){
                Tx = pos_xyz[0] + x*map_resloution - length/2;
                for(int i = 0; i < num_scan_points; i++){
                    cloud_test_loc[i][0] = cloud_test_rot[i][0] + Tx;
                }

                for(int y = 0; y < n_steps; y++){
                    Ty = pos_xyz[1] + y*map_resloution - length/2;
                    for(int i = 0; i < num_scan_points; i++){
                        cloud_test_loc[i][1] = cloud_test_rot[i][1] + Ty;
                    }

                    // will not be needed for the robot... as z is constant 
                    for(int z = 0; z < n_steps; z++){
                        Tz = pos_xyz[2] + z*map_resloution - length/2;
                        sum_ = 0;
                        for(int i = 0; i < num_scan_points; i++){
                            cloud_test_loc[i][2] = cloud_test_rot[i][2] + Tz;
                             
                            indx[0] = round(cloud_test_loc[i][0]*map_conv);
                            indx[1] = round(cloud_test_loc[i][1]*map_conv);
                            indx[2] = round(cloud_test_loc[i][2]*map_conv);

                            if (indx[0] < 0 || indx[0] >= map_grid_size_x){
                                continue;
                            }
                            if (indx[1] < 0 || indx[1] >= map_grid_size_y){
                                continue;
                            }
                            if (indx[2] < 0 || indx[2] >= map_grid_size_z){
                                continue;
                            }
                            sum_ += map_localization[indx[0]][indx[1]][indx[2]];

                        }

                        if (sum_ > sum_prob){
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

        euler_zyx_diff_scan[0] = rot_z_loc - euler_zyx[0];
        pos_xyz_diff_scan[0] = Tx_loc - pos_xyz[0];
        pos_xyz_diff_scan[1] = Tx_loc - pos_xyz[1];
        pos_xyz_diff_scan[2] = Tx_loc - pos_xyz[2];
    }

    void scan_localization_map_update(){
        // this should run after sensor fusion
        // delete old points in loc map
        for(int i = 0; i < num_of_old_points; i++){
            if (points_scan_map_life[i] == 1 || points_scan_map_life[i] == 0){
                if (points_scan_map_life[i] != 0){
                    for(int j = i*res_h*res_v; j < i*res_h*res_v + points_scan_map_index[i]; j++){
                        map_localization[points_scan_map[j][0]][points_scan_map[j][1]][points_scan_map[j][2]] = 0;
                    }
                }
                
                rotate_point(cloud_test_rot, cloudpoints, cloudpoints_index, rot_z, rot_y, rot_x)
                // caclulate all index for new points.


                // add index to list at place and update points_scan_map_index

                // add 6 to life

                // remove 1 life 



            }
        }      
    }
    */
   


    void rotate_point(double new_point[][3], double point[][3], int num_points, double z, double y, double x){
        // need to be implemented...
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


};

void publish_cloud_points(const Slam& slam, ros::Publisher& pub, uint32_t seq){
    // creates pointcloud message and publish it. 
    if(slam.publish_cloud == 1){
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

        unsigned int packed_int_rgb;
        float packed_float_rgb;

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
        msg.channels.push_back(point_rgb);
        pub.publish(msg);



    }

}


int main(int argc, char **argv){

    Slam slam;

    ros::init(argc, argv, "Slam");

    ros::NodeHandle n;

    ros::Publisher publish_point_cloud = n.advertise<sensor_msgs::PointCloud>("PointCloud", 1);

    ros::Subscriber sub = n.subscribe("/camera/depth/image_rect_raw", 10, &Slam::callback, &slam);

    ros::Rate loop_rate(10);

    uint32_t seq = 1;

    while (ros::ok()){
        publish_cloud_points(slam, publish_point_cloud, seq);
        ros::spinOnce();
        loop_rate.sleep();
        ++seq;
    }
    ros::spin();

    return 0;

}






#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud.h"
#include "geometry_msgs/Point32.h"
#include "sensor_msgs/ChannelFloat32.h"
#include <signal.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h> 

#include <random>
#include <iostream> // for cout
#include <chrono>  // for timer
using namespace std;

// globals as not allowed to define in class :(
const static double map_resloution = 0.05; // in meters
const double pi = 2*acos(0.0);


// map grid resolution 0.05 meters.
const static int map_grid_size_x = 10/0.05;// in meters
const static int map_grid_size_y = 10/0.05; // in meters
const static int map_grid_size_z = 4/0.05; // in met
// global heap allocated memmory...
int map_elements = map_grid_size_x*map_grid_size_y*map_grid_size_z;
uint8_t* map_localization = new uint8_t[map_grid_size_x*map_grid_size_y*map_grid_size_z](); // ()initilizes to zero, hopefully. 
// converion x*map_grid_size_y*map_grid_size_z + y*map_grid_size_z + z

int index_conv(int x, int y, int z){
    return x*map_grid_size_y*map_grid_size_z + y*map_grid_size_z + z;
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


        scan_localization();
        scan_localization_map_update();

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
        cout << "Fps "<< fps << " rot " << euler_zyx[0] << " x "<< pos_xyz[0] <<" y "<< pos_xyz[1]<<" z "<< pos_xyz[2] << "\n";


        publish_cloud = 1; // will be moved to somewhere else when the map has been implemented 
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
    double rot_x = 0; // will be changeds to be from gyr acc fitler 
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

        euler_zyx_diff_scan[0] = rot_z_loc - euler_zyx_scan[0];
        pos_xyz_diff_scan[0] = Tx_loc - pos_xyz_scan[0];
        pos_xyz_diff_scan[1] = Ty_loc - pos_xyz_scan[1];
        pos_xyz_diff_scan[2] = Tz_loc - pos_xyz_scan[2];

        euler_zyx_scan[0] = rot_z_loc;
        pos_xyz_scan[0] = Tx_loc;
        pos_xyz_scan[1] = Ty_loc;
        pos_xyz_scan[2] = Tz_loc;
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
                    +abs(pos_xyz_scan[2] - pos_xyz_update[2]) + 30*abs(euler_zyx_scan[0]-euler_zyx_update[0]);


        if (update_value > 15 || first_run == 1){

            
            for(int i = 0; i < num_of_old_points; i++){
                if (points_scan_map_life[i] == 1 || points_scan_map_life[i] <= 0){
                    if (points_scan_map_life[i] != 0){
                        for(int j = i*res_h*res_v; j < i*res_h*res_v + points_scan_map_index[i]; j++){
                            map_localization[points_scan_map[j]] = 0;
                        }
                    }
                    
                    rotate_point(cloudpoints_rot_update, cloudpoints, cloudpoints_index, euler_zyx_scan[0], euler_zyx_scan[1], euler_zyx_scan[2]);
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
            cout << "update" << "\n"; 
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

void mySigintHandler(int sig){
    // do pre shutdown, like delete heap memmory..
    delete[] map_localization; 
    // does result in a segmentation fault as other 
    //thredes will look for this at the moment of shutdown. 

    //
    cout << " Shutdown initilized " << "\n";

    ros::shutdown();
}


int main(int argc, char **argv){

    Slam slam;

    ros::init(argc, argv, "Slam", ros::init_options::NoSigintHandler);

    ros::NodeHandle n;

    signal(SIGINT, mySigintHandler);

    ros::Publisher publish_point_cloud = n.advertise<sensor_msgs::PointCloud>("PointCloud", 1);

    ros::Subscriber sub = n.subscribe("/camera/depth/image_rect_raw", 10, &Slam::callback, &slam);

    /*
    new code here


    */

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






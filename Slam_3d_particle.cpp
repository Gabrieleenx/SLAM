#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud.h"
#include "geometry_msgs/Point32.h"
#include "sensor_msgs/ChannelFloat32.h"

#include <iostream> // for cout
#include <chrono>  // for timer
using namespace std;

/*void print_array(int num_elements, u_int16_t array_[480][640]){
  for (int i = 0; i < num_elements; i++){
    for (int j = 0; j < num_elements; j++){
      cout << unsigned(array_[i][j]) << " ";
    }
    cout << "\n";
  }
  cout << "\n";
}*/

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

    // resize depth image uint8_t to array int by mean over 4x4 and igoring zeors
    void resize_depth_to_array(const vector<uint8_t> &image){
        cout << "inside resize" << "\n";
        for(int i = 0; i < res_v; i++){
            // cout << "h," << i ;
            for(int j = 0; j < res_h; j++){
                //cout << "h," << i << "," << j;
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
                //data little endian but looks like the convertion whants big endian...
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

    //cout<< "kind working"<< unsigned(seq) << "\n";
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
        //cout<< "spin once"<< "\n";
        publish_cloud_points(slam, publish_point_cloud, seq);
        ros::spinOnce();
        loop_rate.sleep();
        ++seq;
    }
    ros::spin();

    return 0;

}






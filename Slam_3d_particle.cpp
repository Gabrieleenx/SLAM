#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"


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

        //cout << "inside callback " << "\n";
        resize_depth_to_array(msg->data);

        img_to_cloud();

        // check computation time
        chrono::high_resolution_clock::time_point time_now = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(time_now - time_last);
        //time_last = time_now;
        int time = duration.count();
        double fps = 1.0/time *1000000;
        cout << "Fps "<< fps << " " << depth_y[31][22] << " "<< depth_x[31][22] <<" "<< depth_z[31][22] << "\n";
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
        int8_index = (z+1)*(x+1)*32-32; // idex in one dimetional vector
        for(int i = 0; i < 4;  i++){
            int8_idx_l = int8_index + i*1280; // 640*2       
            for(int j = 0; j < 4; j++) {
                int8_to_int = (image[int8_idx_l+1] << 8 )| image[int8_idx_l]; // int16 to int, 
                //data little endian but looks like the convertion whats big endian...
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




int main(int argc, char **argv){

    Slam slam;

    ros::init(argc, argv, "listener");

    ros::NodeHandle n;

    ros::Subscriber sub = n.subscribe("/camera/depth/image_rect_raw", 10, &Slam::callback, &slam);

    ros::spin();

    return 0;

}





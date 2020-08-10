#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"

#include <iostream>
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
    int depth_z[480/4][640/4];
    int depth_y[480/4][640/4];
    int depth_x[480/4][640/4];
    

    Slam(){ 
        // constructor 
    }
    
    
    void callback(const sensor_msgs::Image::ConstPtr& msg){
        resize_depth_to_array(msg->data);

    }

private:

    // variables for resize_depth_to_array
    int new_size_x = 640/4; // 160
    int new_size_z = 480/4; // 120 
    // resize depth image uint8_t to array int by mean over 4x4 and igoring zeors
    void resize_depth_to_array(const vector<uint8_t> &image){
        for(int i = 0; i < new_size_z; i++){
            for(int j = 0; j < new_size_x; i++){
                
            }
        }
    }

    // variables for calc_mean_16
    int int8_to_int;
    int int8_index;
    int int8_idx_l;
    // calulates the mean of a 4x4 square from a image and ignores 0 zeros.
    int calc_mean_16(const vector<uint8_t> &image, int z, int x){
        int8_index = (z+1)*(x+1)*32-32;
        for(int i = 0; i < 4;  i++){
            int8_idx_l = int8_index + i*1280; // 640*2       
            for(int j = 0; j < 4; j++) {
                int8_idx_l += j;
                int8_to_int = (image[int8_idx_l] << 8 )| image[int8_idx_l+1];
                // left todo is conditional statement and mean
            }
        }
        return 1;
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






import rclpy
from rclpy.node import Node
import cv2
import os
import numpy as np
import datetime
import time
import sys
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import math

import image_converter as ic
import const
from config import Config
import config 
from sensor_msgs.msg import Joy

config = Config.data_collection
if config['vehicle_name'] == 'broverette':
    from ackbot_msgs.msg import Control 
else:
    exit(config['vehicle_name'] + ' not supported vehicle.')


class DataCollection(Node):
    def __init__(self):
        super().__init__('data_collection_node')
        self.stop_script = False  

        self.steering = 0
        self.throttle = 0
        self.brake = 0

        self.vel = 0
        self.vel_x = self.vel_y = self.vel_z = 0
        self.ang_vel_x = self.ang_vel_y = self.ang_vel_z = 0
        self.pos_x = self.pos_y = self.pos_z = 0

        self.img_cvt = ic.ImageConverter()

        # Create CSV data file
        name_datatime = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        path = 'e2e_data' + '/' + sys.argv[1] + '/' + name_datatime + '/'
                        
        if os.path.exists(path):
            self.get_logger().info('The path exists. Continuing...')
        else:
            self.get_logger().info(f'A new folder created: {path}')
            os.makedirs(path)

        self.text = open(str(path) + name_datatime + const.DATA_EXT, "w+")
        self.text.write(const.DATA_HEADER)

        self.path = path

        # Subscriptions
        self.create_subscription(Control, config['vehicle_control_topic'], self.steering_throttle_cb, 10)
        self.create_subscription(Odometry, config['base_pose_topic'], self.pos_vel_cb, 10)
        self.create_subscription(Image, config['camera_image_topic'], self.recorder_cb, 10)
        self.create_subscription(Joy, 'joy', self.joy_callback, 10)  

    def calc_velocity(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)

    def steering_throttle_cb(self, value):
        self.throttle = value.throttle
        self.steering = value.steer
        self.brake = value.brake

    def pos_vel_cb(self, value):
        self.pos_x = value.pose.pose.position.x 
        self.pos_y = value.pose.pose.position.y
        self.pos_z = value.pose.pose.position.z

        self.vel_x = value.twist.twist.linear.x 
        self.vel_y = value.twist.twist.linear.y
        self.vel_z = value.twist.twist.linear.z
        self.vel = self.calc_velocity(self.vel_x, self.vel_y, self.vel_z)

        self.ang_vel_x = value.twist.twist.angular.x 
        self.ang_vel_y = value.twist.twist.angular.y
        self.ang_vel_z = value.twist.twist.angular.z

    def recorder_cb(self, msg):
        # Convert the ROS2 Image message to a numpy array
        yuyv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 2))

        # Convert YUYV to BGR format
        rgb_image = cv2.cvtColor(yuyv_img, cv2.COLOR_YUV2BGR_YUYV)

        unix_time = time.time()
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        file_full_path = str(self.path) + str(time_stamp) + const.IMAGE_EXT

        cv2.imwrite(file_full_path, rgb_image)
        sys.stdout.write(file_full_path + '\r')

        line = "{}{},{:.4f},{:.4f},{:.4f},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\r\n".format(
                                                time_stamp, const.IMAGE_EXT, 
                                                self.steering, 
                                                self.throttle,
                                                self.brake,
                                                unix_time,
                                                self.vel,
                                                self.vel_x,
                                                self.vel_y,
                                                self.vel_z,
                                                self.ang_vel_x,
                                                self.ang_vel_y,
                                                self.ang_vel_z,
                                                self.pos_x,
                                                self.pos_y,
                                                self.pos_z)

        self.text.write(line)

        cv2.imshow("Camera Feed", rgb_image)

        if cv2.waitKey(1) != -1: 
            self.get_logger().info('Key pressed, stopping the feed and closing the script.')
            self.stop_script = True  

    def joy_callback(self, message):
        """
        Handles incoming Joy messages to stop the script when Button 2 is pressed.
        """
        if message.buttons[2] == 1:  
            self.get_logger().info('Button 2 pressed, stopping the feed and closing the script.')
            self.stop_script = True  

def main(args=None):
    rclpy.init(args=args)
    
    dc = DataCollection()

    try:
        while rclpy.ok() and not dc.stop_script:  
            rclpy.spin_once(dc)
    except KeyboardInterrupt:
        pass

    
    dc.text.close()
    cv2.destroyAllWindows()  
    rclpy.shutdown()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: ')
        exit('$ ros2 run data_collection data_collection.py your_data_id')

    main()

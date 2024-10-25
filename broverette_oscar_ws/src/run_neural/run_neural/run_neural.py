#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import threading
import cv2
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import sys
import os

import const
from image_converter import ImageConverter
from drive_run import DriveRun
from config import Config
from image_process import ImageProcess
import gpu_options


config = Config.neural_net


if Config.data_collection['vehicle_name'] == 'broverette':
    from ackbot_msgs.msg import Control 
else:
    exit(Config.data_collection['vehicle_name'] + ' not supported vehicle.')


class NeuralControl(Node):
    def __init__(self, weight_file_name, weight_file_name2=None):
        super().__init__('run_neural')

        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.drive = DriveRun(weight_file_name)
        if weight_file_name2:
            self.drive2 = DriveRun(weight_file_name2)  # Use multiple network models if available
        else:
            self.drive2 = None

        self.image = None
        self.image_processed = False
        self.braking = False
        self.velocity = 0
        self.rate = self.create_rate(30)

    def _controller_cb(self, image):
        # Process the incoming image data
        yuyv_img = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 2))

        # Convert YUYV to RGB
        img = cv2.cvtColor(yuyv_img, cv2.COLOR_YUV2BGR_YUYV)

        

        # Crop and resize the image as per the neural network input requirements
        cropped = img[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                      Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
        img = cv2.resize(cropped, (config['input_image_width'], config['input_image_height']))

        # Pass the processed image to the neural network
        self.image = self.image_process.process(img)

        if config['lstm']:
            self.image = np.array(self.image).reshape(1, 
                                config['input_image_height'],
                                config['input_image_width'],
                                config['input_image_depth'])

        self.image_processed = True

    def pos_vel_cb(self, value):
        # Update the robot's velocity from odometry data
        vel_x = value.twist.twist.linear.x
        vel_y = value.twist.twist.linear.y
        vel_z = value.twist.twist.linear.z
        self.velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

    def control_loop(self):
        joy_data = Control()
        use_predicted_throttle = config['num_outputs'] == 2

        # Setup subscriptions for image and odometry topics
        self.create_subscription(Image, Config.data_collection['camera_image_topic'], self._controller_cb, QoSProfile(depth=10))
        self.create_subscription(Odometry, Config.data_collection['base_pose_topic'], self.pos_vel_cb, QoSProfile(depth=10))

        # Publisher for control messages
        self.joy_pub = self.create_publisher(Control, Config.data_collection['vehicle_control_topic'], 10)

        print('Start running. Vroom. Vroom. Vroooooom......')
        print('steer \tthrottle \tbrake \tvelocity')

        while rclpy.ok():
            if not self.image_processed:
                rclpy.spin_once(self)
                continue

            # Predicted steering and throttle
            if config['num_inputs'] == 2:
                prediction = self.drive.run((self.image, self.velocity))
            else:
                prediction = self.drive.run((self.image,))

            steer = prediction[0][0]
            throttle = prediction[0][1] if use_predicted_throttle else Config.run_neural['throttle_default']

            # Create control message
            joy_data.steer = float(steer)
            joy_data.throttle = float(throttle)
            joy_data.brake = 0.0  # No brake applied unless needed
            joy_data.shift_gears = Control.FORWARD  # Always in forward gear for now

            # Logic for sharp turns or excessive speed
            if abs(joy_data.steer) > Config.run_neural['sharp_turn_min'] or self.velocity > Config.run_neural['max_vel']:
                joy_data.throttle = Config.run_neural['throttle_sharp_turn']
                joy_data.brake = Config.run_neural['brake_val']
                self.apply_brake()

            # Publish the control message
            self.joy_pub.publish(joy_data)

            # Display the image
            cv2.imshow('Processed Image', self.image)
            key = cv2.waitKey(1)  # Capture key press

            # If any key is pressed, send Neutral gear shift and break the loop
            if key != -1:  # Key is pressed
                print("Key pressed! Shifting to Neutral...")
                joy_data.shift_gears = Control.NEUTRAL  # Send neutral command
                self.joy_pub.publish(joy_data)
                break  # Exit the loop

            # Log the current control values
            cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f}\r'.format(
                joy_data.steer, joy_data.throttle, joy_data.brake, self.velocity
            )
            sys.stdout.write(cur_output)
            sys.stdout.flush()

            # Reset for the next loop
            self.image_processed = False
            self.rate.sleep()

    def apply_brake(self):
        self.braking = True
        timer = threading.Timer(Config.run_neural['brake_apply_sec'], self._timer_cb)
        timer.start()

    def _timer_cb(self):
        self.braking = False


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 2:
        print('Usage:\n$ ros2 run run_neural run_neural weight_file_name [weight_file_name2]')
        sys.exit(1)

    weight_file_name = sys.argv[1]
    weight_file_name2 = sys.argv[2] if len(sys.argv) > 2 else None

    neural_control = NeuralControl(weight_file_name, weight_file_name2)
    try:
        neural_control.control_loop()
    except KeyboardInterrupt:
        print("\nShutdown requested. Exiting...")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Sep 23 13:23:14 2017
# History:
# 11/28/2020: modified for OSCAR
# 10/06/2024: modified for BROVERETTE & ROS 2 Humble

# @author: jaerock
# Modified by: Humberto
# """

# import threading
# import cv2
# import time
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile

# import numpy as np
# from sensor_msgs.msg import Image
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Twist
# import math
# import sys
# import os

# import const
# from image_converter import ImageConverter
# from drive_run import DriveRun
# from config import Config
# from image_process import ImageProcess
# import gpu_options


# config = Config.neural_net
# velocity = 0

# if Config.data_collection['vehicle_name'] == 'broverette':
#     from ackbot_msgs.msg import Control 
# else:
#     exit(Config.data_collection['vehicle_name'] + ' not supported vehicle.')


# class NeuralControl(Node):
#     def __init__(self, weight_file_name, weight_file_name2=None):
#         super().__init__('run_neural')

#         self.ic = ImageConverter()
#         self.image_process = ImageProcess()
#         self.drive = DriveRun(weight_file_name)
#         if weight_file_name2:
#             self.drive2 = DriveRun(weight_file_name2)  # Use multiple network models if available
#         else:
#             self.drive2 = None

#         self.image = None
#         self.image_processed = False
#         self.braking = False
    

#     def _controller_cb(self, image):
#         yuyv_img = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 2))

#         # Convert the ROS2 Image message to a numpy array
#         # yuyv_img = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 2))

#         # Convert YUYV to BGR format
#         img = cv2.cvtColor(yuyv_img, cv2.COLOR_YUV2RGB_YUYV)

#         cropped = img[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
#                       Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
        
#         img = cv2.resize(cropped, (config['input_image_width'], config['input_image_height']))
#         self.image = self.image_process.process(img)

#         # Handle CNN-LSTM models
#         if config['lstm'] is True:
#             self.image = np.array(self.image).reshape(1, 
#                                  config['input_image_height'],
#                                  config['input_image_width'],
#                                  config['input_image_depth'])

#         self.image_processed = True

#     def _timer_cb(self):
#         self.braking = False

#     def apply_brake(self):
#         self.braking = True
#         timer = threading.Timer(Config.run_neural['brake_apply_sec'], self._timer_cb)
#         timer.start()

#     def pos_vel_cb(self, value):
#         global velocity

#         vel_x = value.twist.twist.linear.x
#         vel_y = value.twist.twist.linear.y
#         vel_z = value.twist.twist.linear.z

#         velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

#     def control_loop(self):
#         joy_data = Control()
#         use_predicted_throttle = True if config['num_outputs'] == 2 else False
#         self.rate = self.create_rate(30)

#         self.create_subscription(Image, Config.data_collection['camera_image_topic'], self._controller_cb, QoSProfile(depth=10))
#         self.create_subscription(Odometry, Config.data_collection['base_pose_topic'], self.pos_vel_cb, QoSProfile(depth=10))

#         # Publishers
#         self.joy_pub = self.create_publisher(Control, Config.data_collection['vehicle_control_topic'], 10)

#         print('\nStart running. Vroom. Vroom. Vroooooom......')
#         print('steer \tthrt: \tbrake \tvelocity')

#         while rclpy.ok():
#             if not self.image_processed:
#                 rclpy.spin_once(self)
#                 continue

#             # Predicted steering angle from the input image
#             steer = 0.0
#             throttle = 0.0

#             if config['num_inputs'] == 2:
#                 prediction = self.drive.run((self.image, velocity))
#                 if self.drive2:
#                     prediction2 = self.drive2.run((self.image, velocity))
#                 if config['num_outputs'] == 2:
#                     steer = prediction[0][0]
#                     throttle = prediction[0][1]
#                 else:
#                     steer = prediction[0][0]
#             else:
#                 prediction = self.drive.run((self.image,))
#                 if self.drive2:
#                     prediction2 = self.drive2.run((self.image,))
#                 if config['num_outputs'] == 2:
#                     steer = prediction[0][0]
#                     throttle = prediction[0][1]
#                 else:
#                     steer = prediction[0][0]

#             joy_data.steer = float(steer)
#             joy_data.throttle = float(throttle)
#             joy_data.shift_gears = Control.FORWARD

#             # Simple controller logic
#             is_sharp_turn = False
#             if not self.braking:
#                 if velocity < Config.run_neural['velocity_0']:
#                     joy_data.throttle = Config.run_neural['throttle_default']
#                     joy_data.brake = 0.0
#                 elif abs(joy_data.steer) > Config.run_neural['sharp_turn_min']:
#                     is_sharp_turn = True

#                 if is_sharp_turn or velocity > Config.run_neural['max_vel']:
#                     joy_data.throttle = Config.run_neural['throttle_sharp_turn']
#                     joy_data.brake = Config.run_neural['brake_val']
#                     self.apply_brake()
#                 else:
#                     if not use_predicted_throttle:
#                         joy_data.throttle = Config.run_neural['throttle_default']
#                     joy_data.brake = 0.0
#             joy_data.throttle = 0.0
#             # Publish control commands
#             self.joy_pub.publish(joy_data)
#             #print(f'{joy_data}')
#             # Print current state
#             cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f}\r'.format(
#                 joy_data.steer, joy_data.throttle, joy_data.brake, velocity
#             )
#             sys.stdout.write(cur_output)
#             sys.stdout.flush()

#             # Prepare for next input
#             self.image_processed = False
#             self.rate.sleep()


# def main(args=None):
#     rclpy.init(args=args)

#     if len(sys.argv) < 2:
#         print('Usage:\n$ ros2 run run_neural run_neural weight_file_name [weight_file_name2]')
#         sys.exit(1)

#     weight_file_name = sys.argv[1]
#     weight_file_name2 = sys.argv[2] if len(sys.argv) > 2 else None

#     neural_control = NeuralControl(weight_file_name, weight_file_name2)
#     print("here..")
#     try:
#         neural_control.control_loop()
#     except KeyboardInterrupt:
#         print("\nShutdown requested. Exiting...")
#     finally:
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()

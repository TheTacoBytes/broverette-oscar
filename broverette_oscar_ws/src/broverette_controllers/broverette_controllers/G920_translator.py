#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from broverette_msgs.msg import Control

#######################################
# Logitech G920 with Pedal and Shift

# Steering
STEERING_AXIS = 0   # left 1 --> center 0 --> right -1

# Throttle and Brake
THROTTLE_AXIS = 1   # release 1 --> press -1
BRAKE_AXIS = 2      # release 1 --> press -1

# Gear shift buttons
FORWARD_GEAR_BUTTON = 0     # Button 0 for forward
REVERSE_GEAR_BUTTON = 1     # Button 1 for reverse
NEUTRAL_GEAR_BUTTON = 2     # Button 2 for neutral

class G920ControlTranslator(Node):
    def __init__(self):
        super().__init__('g920_control_translator')

        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )

        self.publisher = self.create_publisher(Control, 'broverette_control', 10)

        self.last_published = None
        self.last_published_time = self.get_clock().now()

        # Latch state for gear shifting
        self.gear_state = Control.NO_COMMAND

        # Timer to ensure control messages are sent at a constant rate (20 Hz)
        self.create_timer(0.05, self.timer_callback)

    def joy_callback(self, message):
        """
        Handles incoming Joy messages, translates them to Control messages,
        and publishes them.
        """
        # Log raw values for debugging
        self.get_logger().info(f"Raw Throttle: {message.axes[THROTTLE_AXIS]}")
        self.get_logger().info(f"Raw Brake: {message.axes[BRAKE_AXIS]}")
        self.get_logger().info(f"Raw Steering: {message.axes[STEERING_AXIS]}")

        # Remap throttle and brake: [-1, 1] to [1, 0]
        throttle_value = (1 + message.axes[THROTTLE_AXIS]) / 2  # Remap to [0, 1]
        brake_value = (1 + message.axes[BRAKE_AXIS]) / 2        # Remap to [0, 1]
        steering_value = message.axes[STEERING_AXIS]            # Already in range [-1, 1]

        command = Control()

        # Handle brake logic
        command.brake = brake_value

        # Handle throttle logic
        command.throttle = throttle_value

        # Gear shifting logic with latch
        if message.buttons[FORWARD_GEAR_BUTTON] == 1:
            self.gear_state = Control.FORWARD
        elif message.buttons[REVERSE_GEAR_BUTTON] == 1:
            self.gear_state = Control.REVERSE
        elif message.buttons[NEUTRAL_GEAR_BUTTON] == 1:
            self.gear_state = Control.NEUTRAL

        # Set the latched gear state
        command.shift_gears = self.gear_state

        # Steering
        command.steer = steering_value

        # Log the mapped values for debugging
        self.get_logger().info(f"Throttle (mapped): {command.throttle}")
        self.get_logger().info(f"Brake (mapped): {command.brake}")
        self.get_logger().info(f"Steering: {command.steer}")
        self.get_logger().info(f"Gear: {self.gear_state}")

        # Publish the control command
        self.last_published = command
        self.publisher.publish(command)

    def timer_callback(self):
        """
        Ensures that the control messages are republished at a fixed rate (20Hz).
        """
        if self.last_published:
            self.publisher.publish(self.last_published)

def main(args=None):
    rclpy.init(args=args)

    node = G920ControlTranslator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


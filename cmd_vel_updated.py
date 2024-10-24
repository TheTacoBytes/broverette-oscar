def control_callback(self, msg):
    """
    Handle incoming control messages from the fusion topic (Control.msg).
    Map these messages to the Rosmaster's car motion control.
    """
    # Define a "creep" speed to simulate a car in drive mode
    creep_speed = 0.1  # A small constant speed when no throttle is applied
    max_vx = 0.33      # Maximum allowed velocity

    # Initialize forward velocity (vx)
    vx = 0.0

    # Handle gear shifting
    if msg.shift_gears == Control.NEUTRAL:
        # In neutral, the car should not move
        vx = 0.0
        angular = 0.0

    elif msg.shift_gears == Control.FORWARD:
        # Check if throttle is pressed
        if msg.throttle > 0:
            # Throttle controls forward speed
            vx = min(msg.throttle, max_vx)  # Limit forward velocity to max_vx
        # If brake is pressed, reduce the velocity based on brake value
        elif msg.brake > 0:
            # Reduce speed proportionally with brake and limit to max_vx
            vx = max(0.0, min(self.car.get_motion_data()[0] * (1 - msg.brake), max_vx))
        # If neither throttle nor brake is pressed, apply "creep" speed
        else:
            vx = creep_speed  # Small forward velocity

        # Map steering (-1 to 1) to angular range (-0.5 to 0.5 for your system)
        angular = msg.steer * 0.5

    elif msg.shift_gears == Control.REVERSE:
        # In reverse gear, throttle and brake work similarly but with reversed velocity
        if msg.throttle > 0:
            # Throttle controls reverse speed (negative velocity)
            vx = max(-msg.throttle, -max_vx)  # Limit reverse velocity to max_vx
        elif msg.brake > 0:
            # Reduce reverse speed and cap to max_vx
            vx = min(0.0, max(self.car.get_motion_data()[0] * (1 - msg.brake), -max_vx))
        else:
            vx = -creep_speed  # Small reverse velocity

        # Reverse steering (you may or may not want to reverse steering in reverse mode)
        angular = msg.steer * 0.5

    # For Ackbot, vy is assumed to be 0
    vy = 0

    # Set car motion using Rosmaster's set_car_motion method
    self.car.set_car_motion(vx, vy, angular)

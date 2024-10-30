from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from config import Config

config = Config.data_collection

def generate_launch_description():
    name_arg = DeclareLaunchArgument(
        'name', 
        default_value='default_name',  
        description='Name argument passed to the data_collection node'
    )

    # Determine which controller node to launch based on config
    controller_node = None
    if config['controller'] == "Ps4_Controller":
        controller_node = "ds4_control"
    elif config['controller'] == "Xbox_Controller":
        controller_node = "xbox_control"
    elif config['controller'] == "Logitech_G920":
        controller_node = "g920_control"
    else:
        raise ValueError(f"Unsupported controller type: {config['controller']}")

    return LaunchDescription([
        name_arg,
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen'
        ),
        Node(
            package='broverette_controllers',
            executable=controller_node,  
            name=controller_node,
            output='screen'
        ),
        Node(
            package='data_collection',
            executable='data_collection',
            name='data_collection',
            output='screen',
            arguments=[LaunchConfiguration('name')] 
        )
    ])

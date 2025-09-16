import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # context_adaptation
    pkg_dir = get_package_share_directory('context_adaptation')
    param_file = os.path.join(pkg_dir, 'config', 'salon_dino.yaml')

    cost_publisher = Node(
        package='context_adaptation',
        executable='cost_publisher',
        parameters=[param_file],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(cost_publisher)

    return ld    

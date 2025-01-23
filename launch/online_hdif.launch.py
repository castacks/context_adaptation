import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # visual_mapping_dir = get_package_share_directory('physics_atv_visual_mapping')
    # visual_mapping_launcher = os.path.join(visual_mapping_dir, 'launch', 'dino_localmapping.launch.py')

    return LaunchDescription([       
        # # visual mapping
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(visual_mapping_launcher)
        # ),

        # Declare the use_sim_time argument
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        ),

        # context adaptation
        Node(
            package="context_adaptation",
            executable="dino_costmap_gp_speed_input_output",
            name="context_clustering",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "config_file": "costmap_configs/GP_base.yaml",
                "cost_topic": "/novatel/imu/data",
                "odom_topic": "/superodometry/integrated_to_init",
                "gridmap_topic": "/dino_gridmap",
                "costmap_topic": "/shortrange_costmap",
                "vel_pub_topic": "/controller/target_input",
                "viz": False,
                "pub_anchors": False,
                "pub_stats": True,
            }],
        ),

        # # context adaptation
        # Node(
        #     package="context_adaptation",
        #     executable="cost_publisher",
        #     name="roughness_cost",
        #     output="screen",
        #     parameters=[{
        #         "cost_stats_dir": "cost_configs/wanda_cost_statistics.yaml",
        #         "imu_topic": "/novatel/imu/data",
        #     }],
        # ),
    ])

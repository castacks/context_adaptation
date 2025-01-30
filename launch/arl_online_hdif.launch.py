import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # salon_config_fp_arg = LaunchConfiguration("config_fp")

    # context_adaptation
    pkg_dir = get_package_share_directory('context_adaptation')
    param_file = PathJoinSubstitution([
        pkg_dir, 'config', "salon_dino_warthog_sim.yaml" # salon_config_fp_arg
    ])

    cost_publisher = Node(
        package='context_adaptation',
        executable='cost_publisher_arl_sim',
        parameters=[{
            "use_sim_time": True,
            "config_file": "costmap_configs/GP_base.yaml",
            "cost_stats_dir": "cost_configs/wanda_cost_statistics.yaml",
            "cost_topic": "/warty/vfm_voxels/traversability_cost",
            "cost_array_topic": "/warty/vfm_voxels/traversability_breakdown",
            "cost_baseline_topic": "/warty/vfm_voxels/traversability_cost_baseline",
            "speed_mismatch_topic": "/warty/vfm_voxels/speed_mismatch",
            "imu_topic": "/warty/imu/data",    
            "odom_topic": "/warty/odometry/local",
        }],
        output='screen'
    )

    context_clusterer = Node(
        package='context_adaptation',
        executable='dino_costmap_gp_speed_input_output',
        parameters=[{
            "use_sim_time": True,
            "config_file": "costmap_configs/GP_base.yaml",
            "cost_topic": "/warty/vfm_voxels/traversability_cost", 
            "odom_topic": "/warty/odometry/local",
            "gridmap_topic": "/warty/vfm_voxels/dino_gridmap",
            "costmap_topic": "/warty/map/planning/local",  # "/warty/vfm_voxels/shortrange_costmap",
            "speedmap_topic": "/warty/vfm_voxels/shortrange_speedmap",
            "cvar_speedmap_topic": "/warty/vfm_voxels/hdif_speedmap_cvar",
            "hdif_max_roughness_topic": "/warty/vfm_voxels/hdif_max_roughness",
            "hdif_cvar_topic": "/warty/vfm_voxels/hdif_cvar",
            "vel_pub_topic": "/warty/vfm_voxels/controller/target_input",           
        }],
        output='screen'
    )    

    ld = LaunchDescription()
    ld.add_action(cost_publisher)
    ld.add_action(context_clusterer)


    return ld    

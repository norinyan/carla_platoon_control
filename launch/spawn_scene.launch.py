import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, OpaqueFunction, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node

"""
ros2 launch carla_platoon_control spawn_scene.launch.py
"""
def _make_spawn_action(context):
    pkg_carla_platoon_control = get_package_share_directory('carla_platoon_control')
    pkg_carla_spawn_objects   = get_package_share_directory('carla_spawn_objects')

    vehicles_config_file = PathJoinSubstitution([
        pkg_carla_platoon_control,
        'config',
        'ego_vehicles.json'
    ])

    spawn_objects_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_carla_spawn_objects, 'carla_spawn_objects.launch.py')
        ),
        launch_arguments={
            'objects_definition_file': vehicles_config_file,
            'spawn_sensors_only': 'False',
        }.items()
    )
    return [spawn_objects_launch]


def generate_launch_description():
    host_arg    = DeclareLaunchArgument('host',          default_value='localhost')
    port_arg    = DeclareLaunchArgument('port',          default_value='2000')
    timeout_arg = DeclareLaunchArgument('timeout',       default_value='10')
    town_arg    = DeclareLaunchArgument('town',          default_value='Town04')
    delay_arg   = DeclareLaunchArgument('spawn_delay',   default_value='5.0')
    visualize_arg       = DeclareLaunchArgument('visualize_route',  default_value='false')
    visualize_delay_arg = DeclareLaunchArgument('visualize_delay',  default_value='9.0')

    carla_bridge_node = Node(
        package='carla_ros_bridge',
        executable='bridge',
        name='carla_ros_bridge',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'host': LaunchConfiguration('host'),
            'port': LaunchConfiguration('port'),
            'timeout': LaunchConfiguration('timeout'),
            'town': LaunchConfiguration('town'),
            'ego_vehicle_role_name': ['vehicle_0', 'vehicle_1', 'vehicle_2'],
            'synchronous_mode': True,
            'synchronous_mode_wait_for_vehicle_control_command': True,
            'fixed_delta_seconds': 0.1,
            'register_all_sensors': True,
        }]
    )

    delayed_spawn = TimerAction(
        period=LaunchConfiguration('spawn_delay'),
        actions=[OpaqueFunction(function=_make_spawn_action)]
    )

    visualize_route = TimerAction(
        period=LaunchConfiguration('visualize_delay'),
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    '/home/nor/ros2_carla_ws/src/carla_platoon_control/carla_platoon_control/scenarios/trajectory_generator.py',
                    '--route', 'A',
                ],
                output='screen',
                condition=IfCondition(LaunchConfiguration('visualize_route'))
            )
        ]
    )

    return LaunchDescription([
        host_arg,
        port_arg,
        timeout_arg,
        town_arg,
        delay_arg,
        visualize_arg,
        visualize_delay_arg,
        carla_bridge_node,
        delayed_spawn,
        visualize_route,
    ])

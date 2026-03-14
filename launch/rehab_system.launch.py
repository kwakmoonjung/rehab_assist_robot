import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. 로봇 구동 (roboton 단축어 대체)
    # ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=192.168.1.100 port:=12345 model:=m0609
    dsr_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('dsr_bringup2'), 
            '/launch/dsr_bringup2_rviz.launch.py'
        ]),
        launch_arguments={
            'mode': 'real',
            'host': '192.168.1.100',
            'port': '12345',
            'model': 'm0609'
        }.items()
    )

    # 2. 듀얼 카메라 런치 (지정된 폴더에서 실행)
    # cd ~/cobot_ws/src/cobot2_ws/rehab_assist_robot/object_detection & ros2 launch dual_cam.launch.py
    camera_target_dir = os.path.expanduser('~/cobot_ws/src/cobot2_ws/rehab_assist_robot/object_detection')
    dual_cam_launch = ExecuteProcess(
        cmd=['ros2', 'launch', 'dual_cam.launch.py'],
        cwd=camera_target_dir,
        output='screen'
    )

    # 3. 개별 파이썬 노드들 (ros2 run 명령어들)
    # 현재 디렉토리 구조상 YOLO 모델(yolo11n-pose.pt)이 패키지 루트에 있으므로, 
    # pose_analyzer_all 노드가 모델을 잘 찾을 수 있도록 루트 디렉토리를 cwd로 지정해줍니다.
    workspace_root = os.path.expanduser('~/cobot_ws/src/cobot2_ws/rehab_assist_robot')

    pose_analyzer = Node(
        package='rehab_assist_robot',
        executable='pose_analyzer_all',
        name='pose_analyzer_node',
        output='screen'
    )

    posture_corrector = Node(
        package='rehab_assist_robot',
        executable='posture_corrector_all',
        name='posture_corrector_node',
        output='screen'
    )

    exercise_logger = Node(
        package='rehab_assist_robot',
        executable='exercise_logger_node',
        name='exercise_logger_node',
        output='screen'
    )

    voice_assistant = Node(
        package='rehab_assist_robot',
        executable='voice_assistant',
        name='voice_assistant_node',
        output='screen'
    )

    system_controller = Node(
        package='rehab_assist_robot',
        executable='system_controller',
        name='system_controller_node',
        output='screen'
    )

    user_interface = Node(
        package='rehab_assist_robot',
        executable='user_interface',
        name='user_interface_node',
        output='screen'
    )

    # 4. 모든 실행 객체를 LaunchDescription에 담아 반환
    return LaunchDescription([
        dsr_bringup_launch,
        dual_cam_launch,
        posture_corrector,
        # exercise_logger,
        voice_assistant,
        system_controller,
        pose_analyzer,
        # user_interface
    ])
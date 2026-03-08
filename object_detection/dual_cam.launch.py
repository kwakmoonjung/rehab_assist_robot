import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # realsense2_camera 패키지의 기본 런치 파일 경로 가져오기
    rs_launch_file = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_align_depth_launch.py'
    )

    # 두 카메라에 공통으로 들어갈 해상도 및 기능 파라미터 세팅
    common_args = {
        'depth_module.depth_profile': '848x480x30',
        'rgb_camera.color_profile': '1280x720x30',
        'initial_reset': 'true',
        'align_depth.enable': 'true',
        'enable_rgbd': 'true',
        'pointcloud.enable': 'true'
    }

    # 1. 정면 로봇 카메라 (Robot Cam)
    robot_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_file),
        launch_arguments={
            **common_args,
            'serial_no': '141722076528',
            'camera_namespace': 'robot',
            'camera_name': 'camera'
        }.items()
    )

    # 2. 측면 고정 카메라 (Fixed Cam)
    fixed_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_file),
        launch_arguments={
            **common_args,
            'serial_no': '215322078366',
            'camera_namespace': 'fixed',
            'camera_name': 'camera'
        }.items()
    )

    # 두 카메라를 동시에 실행하도록 반환
    return LaunchDescription([
        robot_cam,
        fixed_cam
    ])
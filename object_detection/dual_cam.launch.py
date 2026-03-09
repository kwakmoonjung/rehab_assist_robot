import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    rs_launch_file = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'examples',
        'align_depth',
        'rs_align_depth_launch.py'
    )

    common_args = {
        'depth_module.depth_profile': '848x480x30',
        'rgb_camera.color_profile': '1280x720x30',
        # 'depth_module.depth_profile': '640x480x30',
        # 'rgb_camera.color_profile': '640x480x30',
        'initial_reset': 'true',
        'align_depth.enable': 'true',
        'enable_rgbd': 'true',
        'pointcloud.enable': 'true'
    }

    # ⭐️ [수정] USB 2.1 연결 카메라용 해상도 설정 (기능은 모두 true로, 데이터량은 최소로!)
    fixed_override_args = {
        'depth_module.depth_profile': '848x480x30', # USB 2.1 한계 타협점 (15 프레임)
        'rgb_camera.color_profile': '1280x720x30',   # USB 2.1 한계 타협점 (15 프레임)
        # 'depth_module.depth_profile': '640x480x30', # USB 2.1 한계 타협점 (15 프레임)
        # 'rgb_camera.color_profile': '640x480x30',   # USB 2.1 한계 타협점 (15 프레임)
        'enable_depth': 'true',          # 요청대로 복구
        'align_depth.enable': 'true',    # 요청대로 복구
        'enable_rgbd': 'true',           # 요청대로 복구
        'pointcloud.enable': 'true'      # 요청대로 복구
    }

    # 1. 정면 로봇 카메라 (Robot Cam - USB 3.0 연결 권장)
    robot_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_file),
        launch_arguments={
            **common_args,
            'serial_no': '_141722076528',
            'camera_namespace': 'robot',
            'camera_name': 'camera'
        }.items()
    )

    # 2. 측면 고정 카메라 (Fixed Cam - USB 2.1 연결)
    fixed_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_file),
        launch_arguments={
            **common_args,
            **fixed_override_args,  # 고정 카메라 해상도 덮어쓰기 적용
            'serial_no': '_147122075430',
            'camera_namespace': 'fixed',
            'camera_name': 'camera'
        }.items()
    )

    return LaunchDescription([
        robot_cam,
        fixed_cam
    ])
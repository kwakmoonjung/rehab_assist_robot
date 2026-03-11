import os
from glob import glob
from setuptools import setup, find_packages  # 🌟 find_packages 임포트 추가!

package_name = 'rehab_assist_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[
        'rehab_assist_robot',
        'robot_control', 
        'voice_processing', 
        'object_detection',
        'database',
        'object_detection.trackers',
    ]),
    data_files=[
        # ROS 2 기본 설정 파일
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # resource 폴더 내 일반 파일들과 숨김 파일(.env) 병합해서 한 번에 복사 (glob 문법 수정)
        (os.path.join('share', package_name, 'resource'), glob('resource/*') + ['resource/.env']),
        
        # launch 폴더 내 런치 파일 복사
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='Rehab Assist Robot Package',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            # 🌟 누락되었던 exercise_logger_node 추가
            'exercise_logger_node = database.exercise_logger_node:main',
            'exercise_planner = voice_processing.exercise_planner:main',
            'posture_corrector_all = robot_control.posture_corrector_all:main',
            'pose_analyzer_all = object_detection.pose_analyzer_all:main',
            'voice_assistant = voice_processing.voice_assistant:main',
            'system_controller = rehab_assist_robot.system_controller:main',
            'user_interface = database.user_interface:main',
        ],
    },
)
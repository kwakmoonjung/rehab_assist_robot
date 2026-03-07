from setuptools import find_packages, setup
import glob

package_name = 'rehab_assist_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[
        'robot_control', 
        'voice_processing', 
        'object_detection'
    ]),

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', glob.glob('resource/*')),
        ('share/' + package_name + '/resource', glob.glob('resource/.env')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_control = robot_control.robot_control:main',
            'robot_control_detect = robot_control.robot_control_detect:main',
            'object_detection = object_detection.detection:main',
            'get_keyword = voice_processing.get_keyword:main',
            'tracking_pose = object_detection.tracking_pose:main',
            'rehab_manager = object_detection.rehab_manager:main',
            'correction_robot = object_detection.correction_robot:main',
            'yolov11n_pose = object_detection.yolov11n_pose:main',
            'yolov11n_pose_space = object_detection.yolov11n_pose_space:main',
            'test1 = object_detection.tracking_pose_Test:main',
            'test2 = voice_processing.get_keyword_Test:main',
        ],
    },
)

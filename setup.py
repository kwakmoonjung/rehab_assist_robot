from setuptools import find_packages, setup
import glob

package_name = 'rehab_assist_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[
        'rehab_assist_robot',
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
            'posture_corrector = robot_control.posture_corrector:main',
            'pose_analyzer = object_detection.pose_analyzer:main',
            'voice_assistant = voice_processing.voice_assistant:main',
            'system_controller = rehab_assist_robot.system_controller:main',
        ],
    },
)

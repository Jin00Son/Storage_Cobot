from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'jetcobot_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetcobot',
    maintainer_email='jinoobluee@gmail.com',
    description='Jetcobot Camera Manipulator Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = jetcobot_pkg.camera_node:main',
            'jetcobot_node = jetcobot_pkg.jetcobot_node:main',
            'task_manager_node = jetcobot_pkg.task_manager_node:main'
        ],
    },
)

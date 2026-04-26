from setuptools import setup
import os
from glob import glob

package_name = 'carla_platoon_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + '.controllers', package_name + '.scenarios'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            glob('carla_platoon_control/config/*')),
        ('share/' + package_name + '/launch',
            glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nor',
    maintainer_email='norin_ubuntu',
    description='Platoon control package for CARLA',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'platoon_node = carla_platoon_control.platoon_node:main',
            'visualizer = carla_platoon_control.visualizer:main',
        ],
    },
)

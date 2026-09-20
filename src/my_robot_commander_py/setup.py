from setuptools import find_packages, setup

package_name = 'my_robot_commander_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cubicdoggo',
    maintainer_email='tinglin194@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cubic_leg1_mujoco_init            = my_robot_commander_py.cubic_leg1_mujoco_init:main',
            'cubic_doggo_mujoco_init           = my_robot_commander_py.cubic_doggo_mujoco_init:main',
            'cubic_doggo_mujoco_ros_bridge     = my_robot_commander_py.cubic_doggo_mujoco_ros_bridge:main',
            'cubic_doggo_mujoco_pin_stand_test = my_robot_commander_py.cubic_doggo_mujoco_pin_stand_test:main',
            'cubic_doggo_mujoco_pin_walk       = my_robot_commander_py.cubic_doggo_mujoco_pin_walk:main',
            'cubic_doggo_mujoco_pin_stand      = my_robot_commander_py.cubic_doggo_mujoco_pin_stand:main',
            'cubic_doggo_mujoco_ppo_stand      = my_robot_commander_py.cubic_doggo_mujoco_ppo_stand:main',
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'broverette_controllers'

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
    maintainer='beto',
    maintainer_email='beto@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'g920_control = broverette_controllers.G920_translator:main',
            'ds4_control = broverette_controllers.Ps4_translator:main',
            'xbox_control = broverette_controllers.Xbox_translator:main',
        ],
    },
)

# Broverette OSCAR Integration


This repository extends the [Broverette](https://github.com/TheTacoBytes/Broverette) project by integrating the OSCAR Deep Neural Network (DNN) for enhanced autonomous driving capabilities. The setup is optimized for **Ubuntu 22** with **ROS 2 Humble** and supports various controllers to facilitate flexible data collection.


## Overview


This project extends the OSCAR architecture with multi-input capabilities and tensor-based training, enhancing processing speed and navigation performance. Additionally, it supports various gaming controllers (DS4, Xbox, and Logitech G920), providing multiple options for data collection control.


### Key Features

- **Multi-Input Model:** Supports images and velocities simultaneously for robust decision-making.
- **Tensor-Based Training:** Faster data processing using TensorFlow tensors.
- **Controller Support:** Configurable for DS4, Xbox, and Logitech G920 controllers.
- **Optimized for Ubuntu 22 and ROS Humble**.
- **Data Collection Launch**: Custom ROS launch files for streamlined data collection with selected controllers.


## Prerequisites

- **Operating System**: Ubuntu 22
- **ROS**: ROS 2 Humble
- **Python Packages**: TensorFlow, Keras, OpenCV
- **Supported Controllers**: DS4, Xbox, or Logitech G920


### GPU Support (Optional)

To enable GPU acceleration with TensorFlow, ensure the following are installed:
- **CUDA**: 11.8
- **cuDNN**: 8.7

You can download CUDA from the [NVIDIA CUDA Toolkit page](https://developer.nvidia.com/cuda-downloads) and cuDNN from the [NVIDIA cuDNN page](https://developer.nvidia.com/cudnn).


## Installation


1. Clone this repository:
   ```
   mkdir -p ~/broverette/b_oscar
   cd ~/broverette/b_oscar
   git clone https://github.com/TheTacoBytes/broverette_oscar .
   ./ros_build.sh
   ```

2. Install the required Python dependencies:
   ``` 
   pip install -r requirements.txt
   ```

3. Configure the controller type in `~/broverette/b_oscar/config/data_collection/broverette-humberto.yaml`:
   ```
   controller: "Ps4_Controller"  # Options: Ps4_Controller, Xbox_Controller, Logitech_G920
   ```


## Usage

### Launching Data Collection


To start data collection with the desired controller, use the following command:
*YOU MUST BE IN THE ROOT OF THE PROJECT*

```
cd ~/broverette/b_oscar/
source setup.bash
ros2 launch data_collection data_collection_launch.py name:=<custom_name>
```

This command initializes:
- `joy_node` for joystick input
- Selected controller node (`ds4_control`, `xbox_control`, or `g920_control`)
- `data_collection` node, which records images, velocities, and other driving measurements
- You must press L1/LB initilize data collection. To pause you can press L1/RB again and you can see a red dot on the screen.
- The robot by default is in neutral gear, X/A will put it in `Drive`, Circle\B will put it in `Reverse`, and Square\X will put it in `Neutral` (order of buttons are PS4\Xbox).
- To stop data collection press Triangle/Y.
- R1\RB will lower the top speed of the car for easier driving. Press again to remove that cap.


### Training the Model


1. Run the training script with your data paths:
   ```
   cd ~/broverette/b_oscar/
   python3 neural_net/train_single.py <data_path1>
   ```

2. Training history and model checkpoints are stored in the `e2e_data/<data_path1>` directory.
3. 1. Run the model script with your data paths:
   ```
   cd ~/broverette/b_oscar/
   ros2 run run_neural run_neural e2e_data/<data_path1>/<weight_file_name>.keras
   ```


## Requirements File


Your `requirements.txt` should look like this:

```
tensorflow[and-cuda]==2.14.0
opencv-python==4.5.5.64
matplotlib==3.5.1
scipy==1.8.0
tqdm==4.66.6
numpy==1.23.5
absl-py==2.2.1
action-msgs==1.2.1
action-tutorials-interfaces==0.20.5
action-tutorials-py==0.20.5
actionlib-msgs==4.8.0
ament-cmake-test==1.3.11
ament-copyright==0.12.12
ament-cppcheck==0.12.12
ament-cpplint==0.12.12
ament-flake8==0.12.12
ament-index-python==1.4.0
ament-lint==0.12.12
ament-lint-cmake==0.12.12
ament-package==0.14.0
ament-pep257==0.12.12
ament-uncrustify==0.12.12
ament-xmllint==0.12.12
angles==1.15.0
annotated-types==0.7.0
appdirs==1.4.4
argcomplete==1.8.1
asn1crypto==1.4.0
asttokens==3.0.0
astunparse==1.6.3
attrs==25.1.0
auditwheel==4.0.0
bagpy==0.5
bcrypt==3.2.0
beautifulsoup4==4.10.0
beniget==0.4.1
bitarray==3.0.0
bitstring==4.3.0
black==24.10.0
blinker==1.4
bond==4.1.2
breezy==3.2.1
Brotli==1.0.9
builtin-interfaces==1.2.1
cachetools==5.5.2
catkin-pkg==1.0.0
certifi==2025.1.31
cffi==1.17.1
cfgv==3.4.0
chardet==4.0.0
charset-normalizer==3.4.1
chart-studio==1.1.0
click==8.0.3
colcon-argcomplete==0.3.3
colcon-bash==0.5.0
colcon-cmake==0.2.29
colcon-common-extensions==0.3.0
colcon-core==0.19.0
colcon-defaults==0.2.9
colcon-devtools==0.3.0
colcon-installed-package-information==0.2.1
colcon-library-path==0.2.1
colcon-metadata==0.2.5
colcon-notification==0.3.0
colcon-output==0.2.13
colcon-override-check==0.0.1
colcon-package-information==0.4.0
colcon-package-selection==0.2.10
colcon-parallel-executor==0.3.0
colcon-pkg-config==0.1.0
colcon-powershell==0.4.0
colcon-python-setup-py==0.2.9
colcon-recursive-crawl==0.2.3
colcon-ros==0.5.0
colcon-test-result==0.3.8
colcon-zsh==0.5.0
colorama==0.4.6
coloredlogs==15.0.1
composition-interfaces==1.2.1
configobj==5.0.6
contourpy==1.3.1
control==0.10.1
control-msgs==4.8.0
controller-manager==2.50.0
controller-manager-msgs==2.50.0
cov-core==1.15.0
coverage==6.2
cryptography==44.0.2
cv-bridge==3.2.1
cycler==0.12.1
dbus-python==1.2.18
decorator==4.4.2
demo-nodes-py==0.20.5
Deprecated==1.2.13
deprecation==2.1.0
diagnostic-msgs==4.8.0
diagnostic-updater==4.0.6
distlib==0.3.9
distro==1.7.0
docutils==0.17.1
domain-coordinator==0.10.0
dulwich==0.20.31
dwb-msgs==1.1.18
empy==3.3.4
example-interfaces==0.9.3
examples-rclpy-executors==0.15.3
examples-rclpy-minimal-action-client==0.15.3
examples-rclpy-minimal-action-server==0.15.3
examples-rclpy-minimal-client==0.15.3
examples-rclpy-minimal-publisher==0.15.3
examples-rclpy-minimal-service==0.15.3
examples-rclpy-minimal-subscriber==0.15.3
exceptiongroup==1.2.2
executing==2.2.0
fastbencode==0.0.5
fasteners==0.14.1
fastimport==0.9.14
fastjsonschema==2.21.1
filelock==3.17.0
flake8==7.1.1
flake8-docstrings==1.7.0
flatbuffers==25.2.10
fonttools==4.56.0
fs==2.4.12
future==0.18.2
gast==0.6.0
geometry-msgs==4.8.0
gnupg==2.3.1
google-api-python-client==1.7.11
google-auth==2.38.0
google-auth-httplib2==0.1.0
google-auth-oauthlib==1.0.0
google-pasta==0.2.0
greenlet==1.1.2
grpcio==1.71.0
guake==3.8.5
h5py==3.13.0
html5lib==1.1
httplib2==0.20.2
humanfriendly==10.0
identify==2.6.6
idna==3.10
image-geometry==3.2.1
imageio==2.25.1
img2pdf==0.4.4
importlib-metadata==4.6.4
importlib_resources==6.5.2
imutils==0.5.4
iniconfig==1.1.1
interactive-markers==2.3.2
ipython==8.32.0
isort==5.13.2
jedi==0.19.2
jeepney==0.7.1
Jinja2==3.0.3
joblib==1.4.2
joint-state-publisher==2.4.0
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
jupyter_core==5.7.2
keras==2.14.0
keyring==23.5.0
kiwisolver==1.4.8
lark==1.1.1
laser-geometry==2.4.0
launch==1.0.8
launch-ros==0.19.9
launch-testing==1.0.8
launch-testing-ros==0.19.9
launch-xml==1.0.8
launch-yaml==1.0.8
launchpadlib==1.10.16
lazr.restfulclient==0.14.4
lazr.uri==1.0.6
libclang==18.1.1
lifecycle-msgs==1.2.1
lockfile==0.12.2
logging-demo==0.20.5
lxml==4.8.0
macaroonbakery==1.3.1
Mako==1.1.3
map-msgs==2.1.0
Markdown==3.7
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.10.1
matplotlib-inline==0.1.7
mccabe==0.7.0
mdurl==0.1.2
mechanize==0.4.7
message-filters==4.3.7
ml-dtypes==0.2.0
mocap4r2-control-msgs==0.0.7
monotonic==1.6
more-itertools==8.10.0
mpi4py==3.1.3
mypy-extensions==1.0.0
namex==0.0.8
nav-2d-msgs==1.1.18
nav-msgs==4.8.0
nav2-common==1.1.18
nav2-msgs==1.1.18
nav2-simple-commander==1.0.0
nbformat==5.10.4
netifaces==0.11.0
networkx==3.0
nodeenv==1.9.1
nose2==0.9.2
numpy==1.23.5
nvidia-cublas-cu11==11.11.3.6
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-nvcc-cu11==11.8.89
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cudnn-cu11==8.7.0.84
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.3.0.86
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusparse-cu11==11.7.5.86
nvidia-nccl-cu11==2.16.5
oauth2client==4.1.3
oauthlib==3.2.2
ocrmypdf==14.0.3
olefile==0.46
opencv-python==4.5.5.64
opt_einsum==3.4.0
optree==0.14.1
osrf-pycommon==2.1.6
packaging==24.2
pandas==2.2.3
paramiko==2.9.3
parso==0.8.4
pathspec==0.12.1
patiencediff==0.2.1
pbr==5.8.0
pcl-msgs==1.0.0
pdfminer.six==20221105
pendulum-msgs==0.20.5
pep8-naming==0.14.1
pexpect==4.8.0
pikepdf==7.1.1
pillow==11.1.0
platformdirs==4.3.6
plotly==5.22.0
pluggy==0.13.0
ply==3.11
pre_commit==4.1.0
prompt_toolkit==3.0.50
protobuf==4.25.6
psutil==5.9.0
ptyprocess==0.7.0
pure_eval==0.2.3
py==1.10.0
py3rosmsgs==1.18.2
pyasn1==0.6.1
pyasn1_modules==0.4.1
pycairo==1.20.1
pycodestyle==2.12.1
pycparser==2.22
pycryptodomex==3.21.0
pycups==2.0.1
pydantic==2.5.2
pydantic_core==2.14.5
pydocstyle==6.1.1
pydot==1.4.2
PyDrive==1.3.1
pyelftools==0.31
pyflakes==3.2.0
pygame==2.6.1
PyGithub==1.55
Pygments==2.19.1
PyGObject==3.42.1
PyJWT==2.3.0
pymacaroons==0.13.0
PyNaCl==1.5.0
pyOpenSSL==25.0.0
pyparsing==3.2.3
PyPDF2==3.0.1
PyQt5==5.15.6
PyQt5-Qt5==5.15.16
PyQt5_sip==12.17.0
pyRFC3339==1.1
pyserial==3.5
pytest==6.2.5
pytest-cov==3.0.0
python-apt==2.4.0+ubuntu4
python-dateutil==2.9.0.post0
python-debian==0.1.43+ubuntu1.1
python-gitlab==2.10.1
python-qt-binding==1.1.2
pythran==0.10.0
pytz==2025.2
pyusb==1.2.1.post1
PyWavelets==1.4.1
pyxdg==0.27
PyYAML==5.4.1
qt-dotgraph==2.2.4
qt-gui==2.2.4
qt-gui-cpp==2.2.4
qt-gui-py-common==2.2.4
quality-of-service-demo-py==0.20.5
rcl-interfaces==1.2.1
rclpy==3.3.16
rcutils==5.1.6
referencing==0.36.2
requests==2.32.3
requests-oauthlib==2.0.0
requests-toolbelt==0.9.1
resource-retriever==3.1.3
retrying==1.3.4
rich==13.9.4
rmw-dds-common==1.6.0
roman==3.3
ros2action==0.18.12
ros2bag==0.15.14
ros2cli==0.18.12
ros2component==0.18.12
ros2doctor==0.18.12
ros2interface==0.18.12
ros2launch==0.19.9
ros2lifecycle==0.18.12
ros2multicast==0.18.12
ros2node==0.18.12
ros2param==0.18.12
ros2pkg==0.18.12
ros2run==0.18.12
ros2service==0.18.12
ros2topic==0.18.12
rosbag2-interfaces==0.15.14
rosbag2-py==0.15.14
rosbags==0.10.7
rosdep==0.26.0
rosdep-modules==0.26.0
rosdistro==1.0.1
rosdistro-modules==1.0.1
rosgraph-msgs==1.2.1
rosidl-adapter==3.1.6
rosidl-cli==3.1.6
rosidl-cmake==3.1.6
rosidl-generator-c==3.1.6
rosidl-generator-cpp==3.1.6
rosidl-generator-py==0.14.4
rosidl-parser==3.1.6
rosidl-runtime-py==0.9.3
rosidl-typesupport-c==2.0.2
rosidl-typesupport-cpp==2.0.2
rosidl-typesupport-fastrtps-c==2.2.2
rosidl-typesupport-fastrtps-cpp==2.2.2
rosidl-typesupport-introspection-c==3.1.6
rosidl-typesupport-introspection-cpp==3.1.6
rospkg==1.6.0
rospkg-modules==1.6.0
rpds-py==0.22.3
rpyutils==0.2.1
rqt==1.1.7
rqt-action==2.0.1
rqt-bag==1.1.5
rqt-bag-plugins==1.1.5
rqt-console==2.0.3
rqt-controller-manager==2.50.0
rqt-dotgraph==0.0.4
rqt-gauges==0.0.3
rqt-graph==1.3.1
rqt-gui==1.1.7
rqt-gui-py==1.1.7
rqt-joint-trajectory-controller==2.45.0
rqt-moveit==1.0.1
rqt-msg==1.2.0
rqt-plot==1.1.5
rqt-publisher==1.5.0
rqt-py-common==1.1.7
rqt-py-console==1.0.2
rqt-reconfigure==1.1.2
rqt-robot-dashboard==0.5.8
rqt-robot-monitor==1.0.6
rqt-robot-steering==1.0.1
rqt-runtime-monitor==1.0.0
rqt-service-caller==1.0.5
rqt-shell==1.0.2
rqt-srv==1.0.3
rqt-tf-tree==1.0.5
rqt-topic==1.5.0
rsa==4.9
ruamel.yaml==0.18.10
ruamel.yaml.clib==0.2.12
scikit-image==0.19.3
scikit-learn==1.6.1
scipy==1.8.0
screen-resolution-extra==0.0.0
seaborn==0.13.2
SecretStorage==3.3.1
sensor-msgs==4.8.0
sensor-msgs-py==4.8.0
setuptools-scm==8.1.0
shape-msgs==4.8.0
six==1.17.0
slam-toolbox==2.6.10
smclib==4.1.2
snowballstemmer==2.2.0
soupsieve==2.3.1
SQLAlchemy==1.4.31
sros2==0.10.6
ssh-import-id==5.11
stack-data==0.6.3
statistics-msgs==1.2.1
std-msgs==4.8.0
std-srvs==4.8.0
stereo-msgs==4.8.0
sympy==1.9
systemd-python==234
teleop-twist-keyboard==2.4.0
tenacity==9.0.0
tensorboard==2.14.1
tensorboard-data-server==0.7.2
tensorflow==2.14.0
tensorflow-estimator==2.14.0
tensorflow-io-gcs-filesystem==0.37.1
tensorrt==8.5.3.1
termcolor==2.5.0
terminator==2.1.1
tesserocr==2.5.2
tf-transformations==1.1.0
tf2-geometry-msgs==0.25.12
tf2-kdl==0.25.12
tf2-msgs==0.25.12
tf2-py==0.25.12
tf2-ros-py==0.25.12
tf2-tools==0.25.12
threadpoolctl==3.6.0
tifffile==2023.2.3
toml==0.10.2
tomli==2.2.1
topic-monitor==0.20.5
tqdm==4.66.6
traitlets==5.14.3
trajectory-msgs==4.8.0
transforms3d==0.4.2
turtlesim==1.4.2
typing_extensions==4.13.0
tzdata==2025.2
ubuntu-drivers-common==0.0.0
ubuntu-pro-client==8001
ufoLib2==0.13.1
ufw==0.36.1
unattended-upgrades==0.1
unicodedata2==14.0.0
unique-identifier-msgs==2.2.1
uritemplate==3.0.1
urllib3==2.3.0
usb-creator==0.3.7
vboxapi==1.0
virtualenv==20.29.1
visualization-msgs==4.8.0
wadllib==1.3.6
wcwidth==0.2.13
webencodings==0.5.1
Werkzeug==3.1.3
wrapt==1.14.1
xdg==5
xkit==0.0.0
zipp==1.0.0
zstandard==0.23.0
```
Good Luck!! RIP 2025 

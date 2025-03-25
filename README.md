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
   python3 neural_net/train.py <data_path1> <data_path2> ...
   ```

2. Training history and model checkpoints are stored in the `train_output` directory.


## Requirements File


Your `requirements.txt` should look like this:

```
tensorflow[and-cuda]==2.14.0
opencv-python==4.5.5.64
matplotlib==3.5.1
scipy==1.8.0
tqdm==4.66.6
numpy==1.23.5
```

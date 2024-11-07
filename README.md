# Broverette OSCAR Integration


This repository contains an adapted version of OSCAR DNN for autonomous driving, customized for the Broverette platform. The setup is optimized for **Ubuntu 22** with **ROS Humble** and supports various controllers to facilitate flexible data collection.


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
   ```bash
   git clone https://github.com/yourusername/broverette_oscar_integration.git
   cd broverette_oscar_integration

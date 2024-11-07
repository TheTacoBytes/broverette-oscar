#!/bin/bash

####
## Assumption: you're at the 'broverette' directory.

##
# oscar main folder location
export BROVERETTE_PATH=$(pwd)

##
# Set up ros and ws
source /opt/ros/humble/setup.bash
source broverette_oscar_ws/install/setup.bash

##
#Multi machine 
EXPORT ROS_DOMAIN_ID=1

##
# add neural_net folder to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/neural_net

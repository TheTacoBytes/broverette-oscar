#!/bin/bash

####
## Assumption: you're at the 'broverette' directory.

##
# oscar main folder location
export BROVERETTE_PATH=$(pwd)

##
# set up catkin_ws with setup.bash
source broverette_ws/install/setup.bash

##
# add neural_net folder to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/neural_net

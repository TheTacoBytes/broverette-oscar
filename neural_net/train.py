import sys
from drive_train import DriveTrain
import os

def train(data_folders, gpu_id):
    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Initialize DriveTrain with the data folders
    drive_train = DriveTrain(data_folders)
    drive_train.train(show_summary=False)

if __name__ == '__main__':
    try:
        # Check if at least one path is provided
        if len(sys.argv) < 2:
            exit('Usage:\n$ python {} data_path [data_path_2 ...] gpu_id_num'.format(sys.argv[0]))

        # If the last argument is numeric, treat it as the GPU ID
        if sys.argv[-1].isdigit():
            gpu_id = sys.argv[-1]
            data_folders = sys.argv[1:-1]
        else:
            gpu_id = "0"  # Default GPU ID
            data_folders = sys.argv[1:]

        # Convert data_folders to a list if only one folder is provided
        if len(data_folders) == 1:
            data_folders = [data_folders[0]]

        # Start training on the combined data from all specified folders
        print(f"Training on combined data from folders: {data_folders} with GPU ID: {gpu_id}")
        train(data_folders, gpu_id)

    except KeyboardInterrupt:
        print('\nShutdown requested. Exiting...')


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import sys
# from drive_train import DriveTrain
# import gpu_options
# import tensorflow as tf
# import os

# def train(data_folders, gpu_id):
#     # Set GPU visibility
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
#     # Initialize DriveTrain with combined data
#     drive_train = DriveTrain(data_folders)  # Modify DriveTrain to accept a list of folders
#     drive_train.train(show_summary=False)

# if __name__ == '__main__':
#     try:
#         # Check for correct usage
#         if len(sys.argv) < 2:
#             exit('Usage:\n$ python {} data_path [data_path_2 ...] gpu_id_num'.format(sys.argv[0]))

#         # Extract GPU ID if it's the last argument
#         if sys.argv[-1].isdigit():
#             gpu_id = sys.argv[-1]
#             data_folders = sys.argv[1:-1]
#         else:
#             gpu_id = "0"  # Default GPU
#             data_folders = sys.argv[1:]

#         # Train a single model on combined data
#         print(f"Training on combined data from folders: {data_folders} with GPU ID: {gpu_id}")
#         train(data_folders, gpu_id)

#     except KeyboardInterrupt:
#         print('\nShutdown requested. Exiting...')


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Sep 23 13:49:23 2017
# History:
# 11/28/2020: modified for OSCAR 

# @author: jaerock
# """


# import sys
# from drive_train import DriveTrain
# import gpu_options
# import tensorflow as tf
# import os

# ###############################################################################
# #
# def train(data_folder_name, gpu_id):
#     #gpu_options.set()

#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

#     drive_train = DriveTrain(data_folder_name)
#     drive_train.train(show_summary = False)

    
# ###############################################################################
# #
# if __name__ == '__main__':
#     try:
#         if (len(sys.argv) == 1):
#             exit('Usage:\n$ python {} data_path gpu_id_num'.format(sys.argv[0]))

#         gpu_id = sys.argv[2] if len(sys.argv) == 3 else "0"
#         train(sys.argv[1], gpu_id)

#     except KeyboardInterrupt:
#         print ('\nShutdown requested. Exiting...')

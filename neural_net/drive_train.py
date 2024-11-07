import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import glob
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.losses import MeanSquaredError

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess
import utilities

config = Config.neural_net

class DriveTrain:
    def __init__(self, data_paths):
        # Timestamp and output folder initialization
        self.timestamp = utilities.get_current_timestamp()
        self.output_folder = f"{self.timestamp}"
        self.trained_model_loc = os.path.join("train_output", self.output_folder)
        os.makedirs(self.trained_model_loc, exist_ok=True)
        self.ckpt_dir = os.path.join(self.trained_model_loc, self.output_folder + '_' + const.CKPT_DIR)
        
        self.image_process = ImageProcess()

        # Prepare data lists
        self.images, self.velocities, self.measurements = [], [], []
        
        # Read CSV and images
        for main_data_path in data_paths:
            csv_files = glob.glob(os.path.join(main_data_path, "*.csv"))
            if not csv_files:
                print(f"Warning: No CSV file found in {main_data_path}")
                continue
            csv_path = csv_files[0]
            drive_data = DriveData(csv_path, self.timestamp)
            drive_data.read()

            # Process each image and measurement
            for image_name, velocity, measurement in zip(drive_data.image_names, drive_data.velocities, drive_data.measurements):
                # print(f"Loading image: {image_name}, Velocity: {velocity}, Measurement: {measurement}")
                full_image_path = os.path.join(main_data_path, image_name)
                image = cv2.imread(full_image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = self._process_image(image)
                    self.images.append(image)
                    self.velocities.append(velocity)
                    self.measurements.append(measurement)
                else:
                    print(f"Warning: Image {full_image_path} could not be loaded.")

        # Print data counts before tensor conversion
        print(f"Total images loaded: {len(self.images)}")
        print(f"Total velocities loaded: {len(self.velocities)}")
        print(f"Total measurements loaded: {len(self.measurements)}")

        # Convert lists to tensors
        self.images = tf.convert_to_tensor(self.images, dtype=tf.float32)
        self.velocities = tf.convert_to_tensor(self.velocities, dtype=tf.float32)
        self.measurements = tf.convert_to_tensor(self.measurements, dtype=tf.float32)

        # Reshape `measurements` based on `num_outputs`
        if config['num_outputs'] == 1:
            # Keep only steering angle as the output
            self.measurements = tf.reshape(self.measurements[:, 0], (-1, 1))
        elif config['num_outputs'] == 2:
            # Include steering angle and throttle
            self.measurements = tf.reshape(self.measurements[:, :2], (-1, 2))

        # Initialize model and other components
        self.net_model = NetModel(data_paths[0])

    def _process_image(self, image):
        # Crop and resize the image, then apply any additional preprocessing
        image = image[Config.neural_net['image_crop_y1']:Config.neural_net['image_crop_y2'],
                      Config.neural_net['image_crop_x1']:Config.neural_net['image_crop_x2']]
        image = cv2.resize(image, (config['input_image_width'], config['input_image_height']))
        return self.image_process.process(image)

    def _prepare_data(self):
        print(f"Number of images loaded: {len(self.images)}")
        print(f"Number of velocities loaded: {len(self.velocities)}")
        print(f"Number of measurements loaded: {len(self.measurements)}")

        # Use tf.data.Dataset to create a dataset
        if config['num_inputs'] == 2:
            # Dual input: images and velocities
            dataset = tf.data.Dataset.from_tensor_slices(((self.images, self.velocities), self.measurements))
        else:
            # Single input: images only
            dataset = tf.data.Dataset.from_tensor_slices((self.images, self.measurements))
        
        # Shuffle and batch the dataset
        dataset = dataset.shuffle(buffer_size=len(self.images))
        
        # Split dataset into training and validation sets
        val_size = int(config['validation_rate'] * len(self.images))
        
        # Separate training and validation datasets
        self.train_data = dataset.take(len(self.images) - val_size).batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)
        self.valid_data = dataset.skip(len(self.images) - val_size).batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)

        print(f"Training data size: {len(list(self.train_data))}")
        print(f"Validation data size: {len(list(self.valid_data))}")

    def _build_model(self, show_summary=True):
        if show_summary:
            self.net_model.model.summary()

    def _start_training(self):
        if not self.train_data:
            raise NameError("Data is not prepared.")

        # Setting up callbacks
        callbacks = []

        # Model checkpoint callback
        if config['checkpoint']:
            model_ckpt_name = os.path.join(
                self.ckpt_dir,
                f"{self.timestamp}_{Config.neural_net_yaml_name}_n{config['network_type']}"
            )
            checkpoint = ModelCheckpoint(
                filepath=model_ckpt_name + "_{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min"
            )
            callbacks.append(checkpoint)

        # Early stopping callback
        earlystop = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=config['early_stopping_patience'],
            verbose=1,
            mode="min"
        )
        callbacks.append(earlystop)

        # TensorBoard callback
        tensorboard_logdir = config['tensorboard_log_dir'] + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=tensorboard_logdir)
        callbacks.append(tensorboard)

        # Start training
        self.train_hist = self.net_model.model.fit(
            self.train_data,
            validation_data=self.valid_data,
            epochs=config['num_epochs'],
            verbose=1,
            callbacks=callbacks
        )

    def _plot_training_history(self):
        plt.figure()
        plt.plot(self.train_hist.history['loss'][1:])
        plt.plot(self.train_hist.history['val_loss'][1:])
        plt.ylabel('MSE Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training Set', 'Validation Set'], loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.trained_model_loc, f"{Config.neural_net_yaml_name}_n{config['network_type']}_model.png"), dpi=150)
        plt.savefig(os.path.join(self.trained_model_loc, f"{Config.neural_net_yaml_name}_n{config['network_type']}_model.pdf"), dpi=150)

    def train(self, show_summary=True):
        self._prepare_data()
        self._build_model(show_summary)
        self._start_training()
        self.net_model.save(os.path.join(self.trained_model_loc, f"{Config.neural_net_yaml_name}_n{config['network_type']}_model"))
        self._plot_training_history()
        Config.summary()

# Entry point
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 train.py <data_path1> <data_path2> ...")
        sys.exit(1)
    
    data_paths = sys.argv[1:]
    drive_train = DriveTrain(data_paths)
    drive_train.train(show_summary=True)

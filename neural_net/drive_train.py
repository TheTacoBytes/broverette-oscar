import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import sklearn
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess
from data_augmentation import DataAugmentation
import utilities

config = Config.neural_net

class DriveTrain:
    def __init__(self, data_paths):
        # Timestamp and output folder initialization
        self.timestamp = utilities.get_current_timestamp()
        self.output_folder = (f"{self.timestamp}")
        self.trained_model_loc = os.path.join("train_output", self.output_folder)
        os.makedirs(self.trained_model_loc, exist_ok=True)
        self.ckpt_dir = os.path.join( self.trained_model_loc,self.output_folder + '_' + const.CKPT_DIR)
        
        self.data_entries = []
        for main_data_path in data_paths:
            csv_files = glob.glob(os.path.join(main_data_path, "*.csv"))
            if not csv_files:
                print(f"Warning: No CSV file found in {main_data_path}")
                continue
            csv_path = csv_files[0]
            drive_data = DriveData(csv_path, self.timestamp)
            drive_data.read()
            # Add entries
            for image_name, velocity, measurement in zip(drive_data.image_names, drive_data.velocities, drive_data.measurements):
                full_image_path = os.path.join(main_data_path, image_name)
                self.data_entries.append((full_image_path, velocity, measurement))

        self.net_model = NetModel(data_paths[0])
        self.image_process = ImageProcess()
        self.data_aug = DataAugmentation()

    def _prepare_data(self):
        samples = self.data_entries
        if config['lstm']:
            self.train_data, self.valid_data = self._prepare_lstm_data(samples)
        else:    
            self.train_data, self.valid_data = train_test_split(samples, test_size=config['validation_rate'])
        
        self.num_train_samples = len(self.train_data)
        self.num_valid_samples = len(self.valid_data)

    def _prepare_lstm_data(self, samples):
        num_samples = len(samples)
        steps = 1
        last_index = (num_samples - config['lstm_timestep']) // steps
        image_names, velocities, measurements = [], [], []

        for i in range(0, last_index, steps):
            sub_samples = samples[i: i + config['lstm_timestep']]
            sub_image_names, sub_velocities, sub_measurements = [], [], []
            for image_name, velocity, measurement in sub_samples:
                sub_image_names.append(image_name)
                sub_velocities.append(velocity)
                sub_measurements.append(measurement)
            image_names.append(sub_image_names)
            velocities.append(sub_velocities)
            measurements.append(sub_measurements)
        
        return train_test_split(list(zip(image_names, velocities, measurements)), test_size=config['validation_rate'], shuffle=False)

    def _generator(self, samples, batch_size=config['batch_size']):
        num_samples = len(samples)
        while True:
            samples = sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images, velocities, measurements = self._prepare_batch_samples(batch_samples)
                X_train = np.array(images).reshape(-1, config['input_image_height'], config['input_image_width'], config['input_image_depth'])
                y_train = np.array(measurements).reshape(-1, 1)
                
                if config['num_inputs'] == 2:
                    X_train_vel = np.array(velocities).reshape(-1, 1)
                    X_train = [X_train, X_train_vel]
                yield X_train, y_train

    def _prepare_batch_samples(self, batch_samples):
        images, velocities, measurements = [], [], []
        for image_name, velocity, measurement in batch_samples:
            print(f"Feeding image to neural network: {image_name}") 
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[Config.neural_net['image_crop_y1']:Config.neural_net['image_crop_y2'],
                          Config.neural_net['image_crop_x1']:Config.neural_net['image_crop_x2']]
            image = cv2.resize(image, (config['input_image_width'], config['input_image_height']))
            image = self.image_process.process(image)
            images.append(image)
            velocities.append(velocity)
            steering_angle, throttle, brake = measurement
            if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                steering_angle = 0
            if config['num_outputs'] == 2:
                measurements.append((steering_angle * config['steering_angle_scale'], throttle))
            else:
                measurements.append(steering_angle * config['steering_angle_scale'])
        return images, velocities, measurements

    def _build_model(self, show_summary=True):
        self.train_generator = self._generator(self.train_data)
        self.valid_generator = self._generator(self.valid_data)
        if show_summary:
            self.net_model.model.summary()

    def _start_training(self):
        from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
        if self.train_generator is None:
            raise NameError("Generators are not ready.")

        # Prepare the checkpoint directory and filename structure
        model_ckpt_name = os.path.join(
            self.ckpt_dir,
            f"{self.timestamp}_{Config.neural_net_yaml_name}_n{config['network_type']}"
        )

        # Setting up callbacks
        callbacks = []

        if config['checkpoint']:
            checkpoint = ModelCheckpoint(
                filepath=model_ckpt_name + "_{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min"
            )
            callbacks.append(checkpoint)

        earlystop = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=config['early_stopping_patience'],
            verbose=1,
            mode="min"
        )
        callbacks.append(earlystop)

        # tensorboard_logdir = os.path.join(self.output_folder, "logs", self.timestamp)
        tensorboard_logdir = config['tensorboard_log_dir'] + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=tensorboard_logdir)
        callbacks.append(tensorboard)

        # Start training
        self.train_hist = self.net_model.model.fit(
            self.train_generator,
            steps_per_epoch=self.num_train_samples // config['batch_size'],
            epochs=config['num_epochs'],
            validation_data=self.valid_generator,
            validation_steps=self.num_valid_samples // config['batch_size'],
            verbose=1,
            callbacks=callbacks,
            use_multiprocessing=True
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
        self.net_model.save(os.path.join(self.trained_model_loc,f"{Config.neural_net_yaml_name}_n{config['network_type']}_model"))
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

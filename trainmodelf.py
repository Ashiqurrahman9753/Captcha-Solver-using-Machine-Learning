import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras

char_to_mask = {
    '<': '_101_',
    '>': '_102_',
    ':': '_103_',
    '"': '_104_',
    '/': '_105_',
    '\\': '_106_',
    '|': '_107_',
    '?': '_108_',
    '*': '_109_',
    '.': '_111_'
}

def encode_special_characters(filename):
    for char, mask in char_to_mask.items():
        filename = filename.replace(char, mask)
    return filename

def decode_special_characters(encoded_filename):
    for char, mask in char_to_mask.items():
        encoded_filename = encoded_filename.replace(mask, char)
    return encoded_filename

def modify_random_image_label(input_string, captcha_length):
    # Check if the input string has less than 7 characters
    if len(input_string) < captcha_length:
        # Add "@" to the string
        modified_string = input_string + "£"
        # Pad the string with "@" to make it 7 characters long
        modified_string = modified_string.ljust(6, "£")
    else:
        # If the input string has 7 or more characters, truncate it to 7 characters
        modified_string = input_string[:captcha_length]
    return modified_string


# Build a Keras model given some parameters
def create_model(captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
  input_tensor = keras.Input(input_shape)
  x = input_tensor
  for i, module_length in enumerate([module_size] * model_depth):
      for j in range(module_length):
          x = keras.layers.Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
          x = keras.layers.BatchNormalization()(x)
          x = keras.layers.Activation('relu')(x)
      x = keras.layers.MaxPooling2D(2)(x)

  x = keras.layers.Flatten()(x)
  x = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name='char_%d'%(i+1))(x) for i in range(captcha_length)]
  model = keras.Model(inputs=input_tensor, outputs=x)

  return model

# A Sequence represents a dataset for training in Keras
# In this case, we have a folder full of images
# Elements of a Sequence are *batches* of images, of some size batch_size
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=numpy.float32)
        y = [numpy.zeros((self.batch_size, len(self.captcha_symbols)), dtype=numpy.uint8) for i in range(self.captcha_length)]

        for i in range(self.batch_size):
            if self.files.keys():
                random_image_label = random.choice(list(self.files.keys()))
                random_image_file = self.files[random_image_label]

                # We've used this image now, so we can't repeat it in this iteration
                self.used_files.append(self.files.pop(random_image_label))

                # We have to scale the input pixel values to the range [0, 1] for
                # Keras so we divide by 255 since the image is 8-bit RGB
                raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                processed_data = numpy.array(rgb_data) / 255.0
                X[i] = processed_data

                # We have a little hack here - we save captchas as TEXT_num.png if there is more than one captcha with the text "TEXT"
                # So the real label should have the "_num" stripped out.
                
                random_image_label = decode_special_characters(random_image_label)
                random_image_label = modify_random_image_label(random_image_label, self.captcha_length)

                for j, ch in enumerate(random_image_label):
                    y[j][i, :] = 0
                    y[j][i, self.captcha_symbols.find(ch)] = 1

        return X, y

def callthis(symbols, batch_size, width, height, length, epochs, output_model_name, input_model, train_dataset, validate_dataset):

    tf.keras.backend.clear_session()
    
    captcha_symbols = None
    with open(symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "No GPU available!"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # with tf.device('/device:GPU:0'):
    with tf.device('/device:GPU:0'):
    # with tf.device('/device:XLA_CPU:0'):
        model = create_model(length, len(captcha_symbols), (height, width, 3))

        if input_model is not None:
            model.load_weights(input_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        model.summary()

        training_data = ImageSequence(train_dataset, batch_size, length, captcha_symbols, width, height)
        validation_data = ImageSequence(validate_dataset, batch_size, length, captcha_symbols, width, height)


        # Create an EarlyStopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',   # Monitor validation loss
            patience=3,           # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True  # Restore model weights from the epoch with the best validation loss
            )
        
        callbacks = [early_stopping,
                     #keras.callbacks.CSVLogger('log.csv'),
                     keras.callbacks.ModelCheckpoint(output_model_name+'.h5', save_best_only=False)]
        
        # Save the model architecture to JSON
        with open(output_model_name+".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit(training_data,
                      validation_data=validation_data,
                      epochs=epochs,
                      callbacks=callbacks,
                      use_multiprocessing=True)
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + output_model_name+'_resume.h5')
            model.save_weights(output_model_name+'_resume.h5')

if __name__ == '__main__':
    callthis("symbols.txt", 64, 128, 64, 6, 20, "model2", None, "C:\Scalable\project_2\preprocess_training", "C:\Scalable\project_2\preprocess_validation")
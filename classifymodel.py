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
import numpy as np
import locale

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:, 0]
    y = [42 if x>=43 else x for x in y]
    return ''.join([characters[x] for x in y])

def callthis(model_name, captcha_dir, output, symbols):
    symbols_file = open(symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with tf.device('/cpu:0'):
        with open(output, 'w') as output_file:
            json_file = open(model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(model_name+'.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])

            list_dir =  os.listdir(captcha_dir)
            
            #locale.setlocale(locale.LC_COLLATE, 'de_DE.UTF-8')
            
            sorted_list_dir = sorted(list_dir, key=lambda x: [ord(c) for c in x])
            
            for x in sorted_list_dir:
                # load image and preprocess it
                raw_data = cv2.imread(os.path.join(captcha_dir, x))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = numpy.array(rgb_data) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w])
                prediction = model.predict(image)
                output_file.write(x + "," + decode(captcha_symbols, prediction) + "\n")

                print('Classified ' + x)

callthis("model2", "C:\Scalable\project_2\preprocessedimg", "stuff.txt", "symbols.txt")
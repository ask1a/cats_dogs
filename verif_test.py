from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/home/ava6210/cats-dogs/sample'
test_dir = os.path.join(base_dir, 'test')

path_model = 'cats_and_dogs_small_1.h5'
model = load_model(path_model)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=1000, class_mode='binary')

for X, y in test_generator:
    print(model.evaluate(X, y, batch_size=50))
    print(model.metrics_names)
    break


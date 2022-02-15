from pathlib import Path
from tensorflow.keras.utils import image_dataset_from_directory as idfd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pathlib import Path
import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
path = Path(dir_path)
dir = str(path.parent.absolute()) + "/all-images"
save_dir = str(path.parent.absolute()) + "/images"

print(dir)

# dir = Path.cwd + "/all-images"
# print(dir)
# print (Path.cwd)

class RetinalImages:
    batch_size = 32
    img_height = 700
    img_width = 605
    train_gen = ImageDataGenerator()
    retinal_data = train_gen.flow_from_directory(directory=dir,
        target_size=(img_height, img_width),
        batch_size=batch_size)
    retinal_data.shape = [img_height, img_width, 3]

# train_ds = idfd(
#   dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)


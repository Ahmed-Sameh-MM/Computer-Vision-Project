from real_data import RealData
from localize import Localize
from recognize import Recognize

if __name__ == '__main__':

    # call read_json to get a json object
    training_images = RealData().read_images_from_json(file_name='training.json')
    Localize().localize_digits(images=training_images, number_of_images=500, show_images=True)
    Recognize().testImages(images=training_images, number_of_images=500)

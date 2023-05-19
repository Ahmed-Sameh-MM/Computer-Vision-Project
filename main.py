from real_data import RealData
from localize import Localize
from recognize import Recognize

if __name__ == '__main__':

    # call read_images_from_json to get the list of image objects
    training_images = RealData().read_images_from_json(file_name='training.json')

    Localize().localize_digits(images=training_images, number_of_images=10, show_images=True)

    Recognize().test_images(images=training_images, number_of_images=10)

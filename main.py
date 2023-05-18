from real_data import RealData
from localize import Localize
from recognize import Recognize

if __name__ == '__main__':

    # call read_json to get a json object
    training_data = RealData().read_json(file_name='training.json')
    Localize().localize_digits(images=training_data)
    Recognize().testImages(images=training_data)



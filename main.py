from real_data import RealData
from localize import Localize


if __name__ == '__main__':

    # call read_json to get a json object
    training_data = RealData().read_json(file_path='training.json')

    for i in range(len(training_data)):

        # retrieving each struct (image info)
        image_struct = training_data[i]

        # editing the filename to be the file path
        image_struct['filename'] = 'train/' + image_struct['filename']

        # updating the value of the struct
        training_data[i] = image_struct

    # printing the length of the json object (digitStruct)
    print(len(training_data))

    # print the number of files in the train folder
    # print(len(os.listdir('train')))

    Localize().localize_digits(training_data=training_data)

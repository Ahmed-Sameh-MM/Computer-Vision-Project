from real_data import RealData
import cv2


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

    # setting the box color to blue
    box_color = (255, 0, 0)

    for struct in training_data:

        # read the path of the image, then assign it to the "imread function"
        image = cv2.imread(struct['filename'])

        # a loop iterating on the bounding boxes of an image, and calculate their starting and end points
        for bounding_box in struct['boxes']:

            # the top left corner
            start = (int(bounding_box['left']), int(bounding_box['top']))

            # the right bottom corner
            end = (int(bounding_box['left']) + int(bounding_box['width']),
                   int(bounding_box['top']) + int(bounding_box['height']))

            # to draw a rectangle on the image
            cv2.rectangle(image, start, end, box_color, 1)

        window_name = 'image'

        cv2.imshow(window_name, image)

        cv2.waitKey(0)

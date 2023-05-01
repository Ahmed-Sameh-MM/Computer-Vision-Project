from real_data import RealData
import cv2
import numpy as np


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

        # convert the image to grayscale format
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to enhance the contrast of the image
        img_equalized = cv2.equalizeHist(img_gray)

        # Apply a median filter with a kernel size of 3
        img_median = cv2.fastNlMeansDenoising(img_equalized)

        kernel_smoothing = np.ones((9, 9), np.float32) / 81
        img_smoothed = cv2.filter2D(img_gray, -1, kernel_smoothing)

        ret, thresh = cv2.threshold(img_median, 125, 255, cv2.THRESH_BINARY)

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # copying the original image to draw contours on it
        img_copy = image.copy()

        cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)

        """
        # a loop iterating on the bounding boxes of an image, and calculate their starting and end points
        for bounding_box in struct['boxes']:

            # the top left corner
            start = (int(bounding_box['left']), int(bounding_box['top']))

            # the right bottom corner
            end = (int(bounding_box['left']) + int(bounding_box['width']),
                   int(bounding_box['top']) + int(bounding_box['height']))

            # to draw a rectangle on the image
            cv2.rectangle(image, start, end, box_color, 1)
        """

        window_name = 'image'
        window_name2 = 'smoothed'

        cv2.imshow(window_name2, img_median)

        cv2.imshow(window_name, img_copy)

        cv2.waitKey(0)

import cv2
import numpy as np


class Localize:
    @staticmethod
    def localize_digits(training_data):
        # setting the box color to blue
        box_color = (255, 0, 0)

        for struct in training_data:

            # read the path of the image, then assign it to the "imread function"
            image = cv2.imread(struct['filename'])

            kernel_smoothing = np.ones((3, 3), np.float32) / 9
            img_smoothed = cv2.filter2D(image, -1, kernel_smoothing)

            # convert the image to grayscale format
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            rgb_planes = cv2.split(img_gray)

            result_planes = []
            result_norm_planes = []
            for plane in rgb_planes:
                dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                bg_img = cv2.medianBlur(dilated_img, 21)
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_8UC1)
                result_planes.append(diff_img)
                result_norm_planes.append(norm_img)

            result = cv2.merge(result_planes)
            result_norm = cv2.merge(result_norm_planes)

            # Apply histogram equalization to enhance the contrast of the image
            img_equalized = cv2.equalizeHist(img_gray)

            # Apply a median filter with a kernel size of 3
            img_median = cv2.fastNlMeansDenoising(result_norm)

            ret, thresh = cv2.threshold(result_norm, 200, 255, cv2.THRESH_BINARY)

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

            window_name = 'result_norm'
            window_name_2 = 'result'

            cv2.imshow(window_name_2, thresh)

            cv2.imshow(window_name, img_copy)

            cv2.waitKey(0)

import cv2
import numpy as np
from constants import *
from typing import List
from image import Image
from result import Result


class Localize:

    images_height = []
    images_width = []

    @staticmethod
    def localize_digits(images: List[Image], number_of_images: int, show_images: bool):

        non_harsh_accuracy = 0
        img1 = cv2.imread('black.png', cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('black.png', cv2.IMREAD_GRAYSCALE)
        accuracy = []

        for i in range(1, number_of_images):
            # read the path of the image, then assign it to the "imread function"
            image = cv2.imread(ROOT_DIR + '/train/' + images[i].filename)
            image_height = int(image.shape[0])
            image_width = int(image.shape[1])
            Localize.images_width.append(image_width)
            Localize.images_height.append(image_height)

            kernel_smoothing = np.ones((3, 3), np.float32) / 9
            #img_smoothed = cv2.filter2D(image, -1, kernel_smoothing)

            # convert the image to grayscale format
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(ROOT_DIR + '/1_gray/' + images[i].filename, img_gray)

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
            result_norm = cv2.merge(result_norm_planes)

            cv2.imwrite(ROOT_DIR + '/2_normalized/' + images[i].filename, result_norm)

            # plt.imshow(result_norm, cmap='gray')
            # plt.axis("off")
            # plt.show()

            # Apply histogram equalization to enhance the contrast of the image
            img_equalized = cv2.equalizeHist(img_gray)

            # Apply a median filter with a kernel size of 3
            img_median = cv2.fastNlMeansDenoising(result_norm)

            ret, thresh = cv2.threshold(result_norm, 200, 255, cv2.THRESH_BINARY)

            cv2.imwrite(ROOT_DIR + '/3_threshold/' + images[i].filename, thresh)


            # plt.imshow(thresh, cmap='gray')
            # plt.axis("off")
            # plt.show()

            # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
            contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            # copying the original image to draw contours on it
            img_copy_for_contours = image.copy()
            img_copy_for_bounding_boxes = image.copy()

            cv2.drawContours(image=img_copy_for_contours, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                             lineType=cv2.LINE_AA)

            cv2.imwrite(ROOT_DIR + '/4_contours/' + images[i].filename, img_copy_for_contours)

            if image_width > 700:
                min_width_contour = 5
                max_width_contour = 90
            elif image_width > 250:

                min_width_contour = 5
                max_width_contour = 60
            else:
                min_width_contour = 4
                max_width_contour = 37

            min_height_contour = 8
            max_height_contour = 150

            for c in contours:
                predicted_box = cv2.boundingRect(c)
                images[i].predicted_bbox.append(predicted_box)
                x = predicted_box[0]
                y = predicted_box[1]
                w = predicted_box[2]
                h = predicted_box[3]
                if min_width_contour < w < max_width_contour and min_height_contour < h < max_height_contour:

                    cv2.rectangle(img_copy_for_bounding_boxes, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # a loop iterating onn the bounding boxes of an image, and calculate their starting and end points
            for bounding_box in images[i].real_bboxes:

                # the top left corner
                start = (int(bounding_box.left), int(bounding_box.top))

                # the right bottom corner
                end = (int(bounding_box.left) + int(bounding_box.width),
                       int(bounding_box.top) + int(bounding_box.height))

                # to draw a rectangle on the image
                cv2.rectangle(img_copy_for_bounding_boxes, start, end, (0, 0, 255), 1)

                images[i].data = img_copy_for_bounding_boxes

            cv2.imwrite((ROOT_DIR + '/output/' + f'{i+1}.png'), img_copy_for_bounding_boxes)

            if show_images:
                cv2.imshow("final result", img_copy_for_bounding_boxes)

                cv2.imshow("black and white after thresh", thresh)

                cv2.imshow("contoured", img_copy_for_contours)

                cv2.waitKey(0)

            ####accuracy calculations#####
            # Draw rectangles on img1 at the locations specified in realOutput.
            for (x, y, w, h) in images[i].predicted_bbox:
                cv2.rectangle(img1, (x, y), (x + w, y + h), 255, 2)

            # Draw rectangles on img2 at the locations specified in myOutput.
            for box in images[i].real_bboxes:
                cv2.rectangle(img2, (int(box.left), int(box.top)), (int(box.left) + int(box.width), int(box.top) + int(box.height)), 255, 3)

            # Use a bitwise AND operation to calculate the intersection of img1 and img2.
            interSection = cv2.bitwise_and(img1, img2)

            # Calculate the IoU percentage.
            accuracy.append((np.sum(interSection == 255) /
                                (np.sum(img1 == 255) + np.sum(img2 == 255) - np.sum(interSection == 255))) * 100)

            # image_accuracy = Accuracy().bb_intersection_over_union(images[i].predicted_bbox, images[i].real_bboxes)
            # non_harsh_accuracy += image_accuracy

        print("Harsh accuracy for localization:", sum(accuracy) / len(accuracy))

        # print("Non Harsh accuracy", non_harsh_accuracy/1000)

        print("max, min", max(Localize.images_width), min(Localize.images_width))
        print("max, min height", max(Localize.images_width), min(Localize.images_width))

        return Result(images=images, accuracy=accuracy)

import math
import cv2

from constants import *

from image import Image
from typing import List


class Recognize:

    @staticmethod
    def ExtractFeatures(img):
        sift = cv2.SIFT_create()
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            keyPoints, descriptors = sift.detectAndCompute(img, None)
            return keyPoints, descriptors
        keyPoints, descriptors = sift.detectAndCompute(gray, None)
        return keyPoints, descriptors

    @staticmethod
    def MatchFeatures(img1, img2):
        try:
            bruteForceMatcher = cv2.BFMatcher()

            keyPoints1, descriptors1 = Recognize.ExtractFeatures(img1)
            keyPoints2, descriptors2 = Recognize.ExtractFeatures(img2)

            matches = bruteForceMatcher.knnMatch(descriptors1, descriptors2, k=2)

            optimizedMatches = []
            for firstImageMatch, secondImageMatch in matches:
                if firstImageMatch.distance < 1 * secondImageMatch.distance:
                    optimizedMatches.append(firstImageMatch)

            similarity_scores = [match.distance for match in optimizedMatches]
            max_distance = max(similarity_scores)
            min_distance = min(similarity_scores)
            normalized_scores = [(max_distance - score) / ((max_distance - min_distance) + 0.0000001) for score in
                                 similarity_scores]

            matched_image = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, optimizedMatches, None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            return sum(normalized_scores) / len(normalized_scores)
        except:
            return math.inf

    @staticmethod
    def testImages(images: List[Image], number_of_images: int):
        # mat = scipy.io.loadmat('train_32x32.mat')
        # X = mat['X']
        # Labels = mat['y']

        reference_image = cv2.imread(ROOT_DIR + '/train/1.png')
        refernece_ratio = (reference_image.shape[1] + reference_image.shape[0]) / 2
        accuracy = []

        for i in range(1, number_of_images):
            image = cv2.imread(ROOT_DIR + '/train/' + images[i].filename)
            image_with_boxes = cv2.imread(ROOT_DIR + '/output/' + images[i].filename)
            image_aspect_ratio = (image.shape[1] + image.shape[0]) / 2
            # labels_count=0
            # for i in range(len(images[i].real_bboxes)):
            #     labels_count += 1
            image_height = int(image.shape[0])
            image_width = int(image.shape[1])
            # threshold of image
            # thresh = ThresholdImage(imgReal)

            for Box in images[i].real_bboxes:
                score = math.inf
                label = str(int(Box.label))
                digit = ""
                img = image[int(Box.top): int(Box.top) + int(Box.height), int(Box.left): int(Box.left) + int(Box.width)]

                for idx, filename in enumerate(os.listdir("digitTemplates")):
                    template = os.path.join("digitTemplates", filename)
                    digitTemplate = cv2.imread(template)

                    desired_height = img.shape[0]
                    aspect_ratio = digitTemplate.shape[1] / digitTemplate.shape[0]
                    desired_width = int(desired_height * aspect_ratio)
                    resized_image = cv2.resize(digitTemplate, (desired_width, desired_height))

                    sim = Recognize.MatchFeatures(resized_image, img)

                    if sim < score:
                        score = sim
                        digit = filename.split(".")[0]
                accuracy.append(digit == label)
                boxA_x1 = Box.left
                boxA_x2 = Box.left + Box.width
                boxA_y1 = Box.top
                boxA_y2 = Box.top + Box.height
                for boxB in images[i].predicted_bbox:
                    boxB_x1 = boxB[0]
                    boxB_x2 = boxB[0] + boxB[2]
                    boxB_y1 = boxB[1]
                    boxB_y2 = boxB[1] + boxB[3]

                    # determine the (x, y)-coordinates of the intersection rectangle
                    x_a = max(boxA_x1, boxB_x1)
                    y_a = max(boxA_y1, boxB_y1)
                    x_b = min(boxA_x2, boxB_x2)
                    y_b = min(boxA_y2, boxB_y2)

                    # compute the area of intersection rectangle
                    inter_area = max(0, int(x_b - x_a + 1)) * max(0, int(y_b - y_a + 1))
                    if inter_area > ((boxA_x2 - boxA_x1) * (boxA_y2 - boxA_y1)) / 2:
                        # Define the text position (above the rectangle)
                        text_position = (int((int(boxA_x1) + int(boxA_x2)) / 2), int(boxA_y1) - 2)
                        # Define the text font
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # Define the text color (BGR format)
                        text_color = (0, 0, 255)
                        # Define the text thickness
                        text_thickness = 1
                        im = cv2.putText(image_with_boxes, str(digit), text_position, font, (image_aspect_ratio / refernece_ratio) * 1, text_color, text_thickness, cv2.LINE_AA)
                        # Display the image
                        # cv2.imshow("Image with Number", im)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        break
            cv2.imwrite((ROOT_DIR + '/final_output/' + f'{i+1}.png'), image_with_boxes)

        print("Number recognition accuracy:", (sum(accuracy) / len(accuracy)) * 100)

    # @staticmethod
    # def match(regions):
    #     method = cv2.TM_CCOEFF  # Choose the template matching method
    #     for region in regions:
    #         for template in Recognize.templates:
    #             resized_template = cv2.resize((Recognize.templates.shape[1], Recognize.templates.shape[0]), template)
    #             result_0 = cv2.matchTemplate(region, resized_template, method)
    #             print(result_0)
    #
    #
    # @staticmethod
    # def extract_rois(images: List[Image]):
    #     for image in images:
    #         # read the path of the image, then assign it to the "imread function"
    #         image = cv2.imread(ROOT_DIR + '/train/' + image.filename)
    #         rois = []
    #         for bbox in image.real_bboxes:
    #             x, y, w, h = bbox.left, bbox.top, bbox.width, bbox.height
    #             roi = image[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates
    #             rois.append(roi)
    #         Recognize.match(roi)



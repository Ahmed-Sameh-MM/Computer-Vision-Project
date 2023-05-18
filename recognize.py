import math
import cv2
import os
from constants import *
from image import Image
from typing import List


class Recognize:
    @staticmethod
    def extract_features(img):
        sift = cv2.SIFT_create()
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            key_points, descriptors = sift.detectAndCompute(img, None)
            return key_points, descriptors
        key_points, descriptors = sift.detectAndCompute(gray, None)
        return key_points, descriptors

    @staticmethod
    def match_features(img1, img2):
        try:
            brute_force_matcher = cv2.BFMatcher()

            key_points1, descriptors1 = Recognize.extract_features(img1)
            key_points2, descriptors2 = Recognize.extract_features(img2)

            matches = brute_force_matcher.knnMatch(descriptors1, descriptors2, k=2)

            optimized_matches = []
            for first_image_match, second_image_match in matches:
                if first_image_match.distance < 1 * second_image_match.distance:
                    optimized_matches.append(first_image_match)

            similarity_scores = [match.distance for match in optimized_matches]
            max_distance = max(similarity_scores)
            min_distance = min(similarity_scores)
            normalized_scores = [(max_distance - score) / ((max_distance - min_distance) + 0.0000001) for score in
                                 similarity_scores]

            matched_image = cv2.drawMatches(img1, key_points1, img2, key_points2, optimized_matches, None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            return sum(normalized_scores) / len(normalized_scores)
        except:
            return math.inf

    @staticmethod
    def test_images(images: List[Image], number_of_images: int):
        reference_image = cv2.imread(ROOT_DIR + '/train/1.png')
        refernece_ratio = (reference_image.shape[1] + reference_image.shape[0]) / 2
        accuracy = []

        for i in range(1, number_of_images):
            image = cv2.imread(ROOT_DIR + '/train/' + images[i].filename)
            image_with_boxes = cv2.imread(ROOT_DIR + '/output/' + images[i].filename)
            image_aspect_ratio = (image.shape[1] + image.shape[0]) / 2
            image_height = int(image.shape[0])
            image_width = int(image.shape[1])

            for Box in images[i].real_bboxes:
                score = math.inf
                label = str(int(Box.label))
                digit = ""
                img = image[int(Box.top): int(Box.top) + int(Box.height), int(Box.left): int(Box.left) + int(Box.width)]

                for idx, filename in enumerate(os.listdir("digitTemplates")):
                    template = os.path.join("digitTemplates", filename)
                    digit_template = cv2.imread(template)

                    desired_height = img.shape[0]
                    aspect_ratio = digit_template.shape[1] / digit_template.shape[0]
                    desired_width = int(desired_height * aspect_ratio)
                    resized_image = cv2.resize(digit_template, (desired_width, desired_height))

                    sim = Recognize.match_features(resized_image, img)

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

                    x_a = max(boxA_x1, boxB_x1)
                    y_a = max(boxA_y1, boxB_y1)
                    x_b = min(boxA_x2, boxB_x2)
                    y_b = min(boxA_y2, boxB_y2)

                    inter_area = max(0, int(x_b - x_a + 1)) * max(0, int(y_b - y_a + 1))
                    if inter_area > ((boxA_x2 - boxA_x1) * (boxA_y2 - boxA_y1)) / 2:
                        text_position = (int((int(boxA_x1) + int(boxA_x2)) / 2), int(boxA_y1) - 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_color = (0, 0, 255)
                        text_thickness = 1
                        im = cv2.putText(image_with_boxes, str(digit), text_position, font,
                                         (image_aspect_ratio / refernece_ratio) * 1, text_color, text_thickness,
                                         cv2.LINE_AA)
                        break
            cv2.imwrite((ROOT_DIR + '/final_output/' + f'{i+1}.png'), image_with_boxes)

        print("Number recognition accuracy:", (sum(accuracy) / len(accuracy)) * 100)

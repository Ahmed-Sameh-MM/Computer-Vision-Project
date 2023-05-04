from box import Box

from typing import List


class Accuracy:

    @staticmethod
    def bb_intersection_over_union(predicted_bbox, real_bbox: List[Box]):

        acc_iou = 0

        for boxA in real_bbox:
            boxA_x1 = boxA.left
            boxA_x2 = boxA.left + boxA.width
            boxA_y1 = boxA.top
            boxA_y2 = boxA.top + boxA.height
            for boxB in predicted_bbox:

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
                inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

                # compute the area of both the prediction and ground-truth
                # rectangles
                box_a_area = (boxA.height + 1) * (boxA.width + 1)
                box_b_area = (boxB[2] + 1) * (boxB[3] + 1)

                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the intersection area
                iou = inter_area / float(box_a_area + box_b_area - inter_area)

                acc_iou += iou

        # return the intersection over union values
        return acc_iou / len(real_bbox)

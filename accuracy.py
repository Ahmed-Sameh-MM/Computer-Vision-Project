from collections import namedtuple


# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])


def bb_intersection_over_union(box1, box2):

    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box1[0], box2[0])
    y_a = max(box1[1], box2[1])
    x_b = min(box1[2], box2[2])
    y_b = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box_b_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou

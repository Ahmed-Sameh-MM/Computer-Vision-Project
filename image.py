from box import Box

from typing import List
import numpy as np


class Image:

    def __init__(self, filename: str, real_bboxes: List[Box]):

        self.filename = filename
        self.real_bboxes = real_bboxes
        self.predicted_bbox = []
        self.data = np.array(0)

    @classmethod
    def from_json(cls, json_str):
        data = json_str
        boxeslist = [Box.from_json(b) for b in data['boxes']]
        return cls(data['filename'], boxeslist)

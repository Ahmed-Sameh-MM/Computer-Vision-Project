from box import Box
from typing import List
import json


class Image:

    def __init__(self, filename: str, real_bboxes: List[Box]):

        self.filename = filename
        self.real_bboxes = real_bboxes
        self.predicted_bbox = []

    @classmethod
    def fromJson(cls, json_str):
        data = json_str
        boxeslist = [Box.fromJson(b) for b in data['boxes']]
        return cls(data['filename'], boxeslist)





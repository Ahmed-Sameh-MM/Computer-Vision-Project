from box import Box
from typing import List
import json


class Image:
    def __init__(self, filename: str, boxes: List[Box]):

        self.filename = filename
        self.boxes = boxes

    @classmethod
    def fromJson(cls, json_str):
        data = json_str
        boxeslist = [Box.fromJson(b) for b in data['boxes']]
        return cls(data['filename'], boxeslist)





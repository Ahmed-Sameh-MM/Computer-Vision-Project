import json
from constants import *
from image import Image
from typing import List



class RealData:

    @staticmethod
    def read_json(file_name: str) -> List[Image]:
        f = open(ROOT_DIR + '/' + file_name)
        data = json.load(f)
        images = [Image.fromJson(i) for i in data]
        return images

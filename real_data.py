import json
from constants import *
from image import Image
from typing import List


class RealData:

    @staticmethod
    def read_images_from_json(file_name: str) -> List[Image]:
        f = open(ROOT_DIR + '/' + file_name)
        data = json.load(f)
        images = [Image.from_json(i) for i in data]
        return images

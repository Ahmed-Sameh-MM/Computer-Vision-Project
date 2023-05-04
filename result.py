from image import Image

from typing import List


class Result:
    def __init__(self, images: List[Image], accuracy: List[float]):
        self.images = images
        self.accuracy = accuracy

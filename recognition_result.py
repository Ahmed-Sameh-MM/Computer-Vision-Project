from image import Image

from typing import List


class RecognitionResult:
    def __init__(self, images: List[Image], accuracy: float):
        self.images = images
        self.accuracy = accuracy

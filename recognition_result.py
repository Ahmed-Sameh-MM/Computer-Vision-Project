from typing import List


class RecognitionResult:
    def __init__(self, matched_images: List, accuracy: float):
        self.matched_images = matched_images
        self.accuracy = accuracy

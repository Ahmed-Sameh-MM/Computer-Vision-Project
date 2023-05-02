import json
from constants import *


class RealData:

    @staticmethod
    def read_json(file_name: str):
        f = open(ROOT_DIR + '/' + file_name)
        data = json.load(f)

        return data

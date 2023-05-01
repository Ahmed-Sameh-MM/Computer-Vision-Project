import json


class RealData:

    @staticmethod
    def read_json(file_path: str):
        f = open(file_path, )
        data = json.load(f)

        # print the first structure (first image data)
        print("The data of the first image is: ", data[0])

        return data

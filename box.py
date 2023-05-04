class Box:

    def __init__(self, height: float, label: float, left: float, top: float, width: float):
        self.height = height
        self.label = label
        self.left = left
        self.top = top
        self.width = width

    @classmethod
    def from_json(cls, json_str):
        data = json_str
        return cls(data['height'], data['label'], data['left'], data['top'], data['width'])

import json

class Conf:
    def __init__(self, confPath):
        conf = json.load(open(confPath))
        self.__dict__.update(conf)

    def __getitem__(self, k):
        return self.__dict__.get(k, None)

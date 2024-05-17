class Config:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.debug = False
        self.test = False

    def set_debug(self, debug):
        self.debug = debug

    def set_test(self, test):
        self.test = test
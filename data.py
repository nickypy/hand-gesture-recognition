import numpy as np

class Data:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def load_data(self):
        return (self.train_x, self.train_y), (self.test_x, self.test_y)


import numpy as np
import scipy as sp

# from test_functions import evaluate


class Point:
    def __init__(self, dim=2, upper_limit=10, lower_limit=-10, objective=None):
        self.dim = dim
        self.coords = np.zeros((self.dim,))
        self.z = None
        self.range_upper_limit = upper_limit
        self.range_lower_limit = lower_limit
        self.objective = objective
        self.evaluate_point()

    def generate_random_point(self):
        self.coords = np.random.uniform(self.range_lower_limit, self.range_upper_limit, (self.dim,))
        self.evaluate_point()
    
    def generate_definite_point(self,cordinates):
        self.coords = np.array(cordinates)
        self.evaluate_point()
    
    def evaluate_point(self):
        # self.z = evaluate(self.coords)
        self.z = self.objective.evaluate(self.coords)


if __name__ == '__main__':
    print("Point class defined in this script")

import copy
import numpy as np
from matplotlib import pyplot as plt

from point import Point
from matplotlib import pyplot as plt

class Population:
    def __init__(self, dim=2, num_points=50, upper_limit=10, lower_limit=-10, init_generate=True, objective=None):
        self.points = []
        self.num_points = num_points
        self.init_generate = init_generate
        self.dim = dim
        self.range_upper_limit = upper_limit
        self.range_lower_limit = lower_limit
        self.objective = objective
        # If initial generation parameter is true, then generate collection
        if self.init_generate == True:
            for ix in xrange(num_points):
                new_point = Point(dim=dim, upper_limit=self.range_upper_limit,
                                  lower_limit=self.range_lower_limit, objective=self.objective)
                new_point.generate_random_point()
                self.points.append(new_point)

    def get_visualization(self, leaders=None, plot_leaders=False):
        # Displays 2D animation of points converging to minima
        pts = self.points
        plt.ion()
        plt.clf()
        plt.xlim(self.range_lower_limit, self.range_upper_limit)
        plt.ylim(self.range_lower_limit, self.range_upper_limit)

        for px in pts:
            plt.scatter(p.coords[0], p.coords[1])

        if plot_leaders == True and leaders is not None:
            for lx in leaders:
                plt.scatter(lx.coords[0], lx.coords[1], c='r')
        plt.pause(0.05)

    def get_average_objective(self):
        avg = 0.0

        for px in self.points:
            px.evaluate_point()
            avg += px.z
        avg = avg/float(self.num_points)
        return avg


if __name__ == '__main__':
    pass
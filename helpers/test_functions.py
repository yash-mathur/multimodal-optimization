import numpy as np

# def evaluate(point):
#     test = Function()
#     return test.rastrigin(point)


class Function:
    def __init__(self, func=None):

        self.objectives = {
            'sphere': self.sphere,
            'ackley': self.ackley,
            'rosenbrock': self.rosenbrock,
            'rastrigin': self.rastrigin,
        }
        
        if func is None:
            self.func_name = 'sphere'
            self.func = self.objectives[self.func_name]
        else:
            if type(func) == str:
                self.func_name = func
                self.func = self.objectives[self.func_name]
            else:
                self.func = func
                self.func_name = func.func_name

    def evaluate(self, point):
        return self.func(point)

    def sphere(self, x):
        d = x.shape[0]
        f = 0.0

        for dx in xrange(d):
            f += x[dx] ** 2
        
        return f

    def ackley(self, x):
        z1, z2 = 0, 0

        for i in xrange(len(x)):
            z1 += x[i] ** 2
            z2 += np.cos(2.0 * np.pi * x[i])

        return (-20.0 * np.exp(-0.2 * np.sqrt(z1 / len(x)))) - np.exp(z2 / len(x)) + np.e + 20.0

    def rosenbrock(self, x):
        v = 0
        for i in xrange(len(x) - 1):
            v += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2

        return v

    def rastrigin(self, x):
        v = 0

        for i in range(len(x)):
            v += (x[i] ** 2) - (10 * np.cos(2 * np.pi * x[i]))

        return (10 * len(x)) + v


if __name__ == '__main__':
    print("A collection of several Test functions for optimizations")

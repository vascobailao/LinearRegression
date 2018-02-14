from LinearRegression import Regression
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')


class MultivariateLR(Regression):

    def __init__(self, x, y, learning_rate, iterations):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.B = np.zeros(self.size)
        self.iterations = iterations

    def cost_function(self):
        cost = np.sum((self.x.dot(self.B) - self.y) ** 2) / (2 * self.size)
        return cost

    def gradient_descent(self):
        cost_history = [0] * self.iterations

        for iteration in range(self.iterations):

            y_hat = self.x.dot(self.B)

            loss = y_hat - self.y

            gradient = self.x.T.dot(loss) / self.size

            self.B = self.B - self.learning_rate * gradient

            cost = self.cost_function()
            cost_history[iteration] = cost

        return self.B, cost_history

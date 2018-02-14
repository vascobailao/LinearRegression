from LinearRegression import Regression
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


class UnivariateLR(Regression):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_params(self):

        first = self.size*np.sum(self.x*self.y) - (np.sum(self.y)*np.sum(self.x))
        second = self.size*np.sum(self.x**2) - (np.sum(self.x))**2

        m = first/second
        b = (np.sum(self.y)-m*np.sum(self.x))/self.size
        return m, b

    def get_residuals(self, m, b):
        self.residuals = np.sum((self.y-self.x*m-b)**2)
        return self.residuals

    def fit_LR(self, m, b):
        return np.array(self.x*m + b)

    def plot_line_train(self, m, b):

        results = self.fit_LR(m, b)
        plt.scatter(self.x, self.y, color="b", s=2)
        plt.plot(self.x, results, '--', color="r")
        plt.show()

    def run(self):

        m, b = self.get_params()
        self.plot_line_train(m, b)


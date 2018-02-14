from LinearRegression import Regression
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


class UnivariateLR(Regression):

    def get_params(self):

        first = self.size*np.sum(self.x*self.y) - (np.sum(self.y)*np.sum(self.x))
        second = self.size*np.sum(self.x**2) - (np.sum(self.x))**2

        m = first/second
        b = (np.sum(self.y)-m*np.sum(self.x))/self.size
        return m, b

    def get_residuals(self):
        self.residuals = np.sum((self.y-self.x*self.m-self.b)**2)
        return self.residuals

    def fit_LR(self):
        return np.array(self.x*self.m + self.b)

    def plot_line_train(self):

        results = self.fit_LR()
        plt.scatter(self.x, self.y, color="b", s=2)
        plt.plot(self.x, results, '--', color="r")
        plt.show()


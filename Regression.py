import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
mpl.use('TkAgg')
import matplotlib.pyplot as plt


class Regression:

    def __init__(self, dic):
        self.dic = dic

    def split_data(self):
        if len(self.dic) == 1:
            msk = np.random.rand(len(self.dic["data"])) < 0.8
            training_data = self.dic["data"][msk]
            test_data = self.dic["data"][~msk]
            return training_data, test_data

        training_data = self.get_trainingData()
        test_data = self.get_testData()
        return training_data, test_data

    def get_size(self, training_data):
        return np.size(training_data)

    def get_shape(self, training_data):
        size = training_data.shape
        return size

    def get_columnNames(self, training_data):
        columns_names = list(training_data.columns.values)
        return columns_names

    def get_trainingData(self):
        if self.dic["train"] is None:
            return None
        df = self.dic["train"]

        return df

    def get_testData(self):
        if self.dic["test"] is None:
            return None
        df = self.dic["test"]
        return df

    def get_validationData(self):
        if self.dic["valid"] is None:
            return None
        df = self.dic["valid"]
        return df

    def get_data(self, columns_names, training_data):
        size = len(columns_names)

        if size > 2:
            independent = np.array(training_data.iloc[:, :size-1])
            dependent = np.array(training_data.iloc[:, size-1])
            return independent, dependent

        independent = np.array(training_data.iloc[:, :size-1]).flatten()
        dependent = np.array(training_data.iloc[:, size-1]).flatten()
        return independent, dependent

    def plot_data(self, training_data, independent, dependent):
        if self.get_shape(training_data)[1] == 1:
            plt.plot(independent, dependent, '.')
            plt.show()

        if self.get_shape(training_data)[1] == 2:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(independent[:, 0], independent[:, 1], dependent, color='#ef1234')
            plt.show()

        else:
            raise ValueError("Data too dimensional to plot")

    def calculate_rmse(self, y, y_hat):
        rmse = np.sqrt(sum((y - y_hat) ** 2) / len(y))
        return rmse

    def calculate_r2(self, y, y_hat):
        mean_y = np.mean(y)
        ss_tot = sum((y - mean_y) ** 2)
        ss_res = sum((y - y_hat) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def run(self, training_data):
        x, y = self.get_data(self.get_columnNames(training_data), training_data)
        if len(self.get_columnNames(training_data)) > 2:
            lr = MultivariateLR(x, y, 0.0001, 10000)
            return lr
        lr = UnivariateLR(x, y)
        return lr


class MultivariateLR(Regression):

    def __init__(self, x, y, learning_rate, iterations):
        self.x = x.T
        self.y = y
        self.size = len(x)
        self.B = np.zeros((self.size,), dtype=int)
        self.learning_rate = learning_rate
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

    def run(self, **kwargs):
        B, cost_history = self.gradient_descent()
        return B, cost_history


class UnivariateLR(Regression):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = len(x)

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

    def run(self, **kwargs):

        m, b = self.get_params()
        return m, b
        #self.plot_line_train(m, b)


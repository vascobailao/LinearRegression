import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from LinearRegression import MultivariateRegression, UnivariateRegression


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
            lr = MultivariateRegression.MultivariateLR(x, y, 0.0001, 10000)
            return lr
        lr = UnivariateRegression.UnivariateLR(x, y)
        return lr






# Created by Vasco B. Fernandes
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
mpl.use('TkAgg')
import matplotlib.pyplot as plt


class Regression:

    def __init__(self, dic):
        self.dic = dic

    '''
    Splits the dictionary in training, testing and validation data set
    
    :returns {Pandas.Dataframe} training_data, test_data
    '''
    def split_data(self):
        if len(self.dic) == 1:
            msk = np.random.rand(len(self.dic["data"])) < 0.8
            training_data = self.dic["data"][msk]
            test_data = self.dic["data"][~msk]
            return training_data, test_data

        training_data = self.get_trainingData()
        test_data = self.get_testData()
        return training_data, test_data

    '''
    Return size of dataset
    
    :param {Pandas.Dataframe} training_data
    :return {np.int64} 
    '''
    def get_size(self, training_data):
        return np.size(training_data)

    '''
    Returns shape of datset
    
    :param {Pandas.Dataframe} training_data
    :return {tuple} size
    '''
    def get_shape(self, training_data):
        size = training_data.shape
        return size

    '''
    Returns a list with column names
    
    :param {Pandas,Dataframe} training_data
    :return {list} column_names
    '''
    def get_columnNames(self, training_data):
        columns_names = list(training_data.columns.values)
        return columns_names

    '''
    Submethod of split data (training set)
    
    :return {Pandas.Dataframe} df
    '''
    def get_trainingData(self):
        if self.dic["train"] is None:
            return None
        df = self.dic["train"]

        return df

    '''
    Submethod of split data (test set)
    :return {Pandas.Dataframe} df
    '''
    def get_testData(self):
        if self.dic["test"] is None:
            return None
        df = self.dic["test"]
        return df

    '''
    Submethod of split data (validation set)
    :return {Pandas.Dataframe} df
    '''
    def get_validationData(self):
        if self.dic["valid"] is None:
            return None
        df = self.dic["valid"]
        return df

    '''
    Divides the training set in independent and dependent sets
    
    :params {list} (column names), {Pandas.Dataframe} training data
    :return {Numpy.Array} independent, dependent (variable)
    '''
    def get_data(self, columns_names, training_data):
        size = len(columns_names)

        if size > 2:
            independent = np.array(training_data.iloc[:, :size-1])
            dependent = np.array(training_data.iloc[:, size-1])
            return independent, dependent

        independent = np.array(training_data.iloc[:, :size-1]).flatten()
        dependent = np.array(training_data.iloc[:, size-1]).flatten()
        return independent, dependent

    '''
    Plots the training data
    
    :param {Pandas.Dataframe} training_data, {Numpy.Array} independent, dependent (variables)
    '''
    def plot_data(self, training_data, independent, dependent):

        if self.get_shape(training_data)[1] == 1:
            plt.plot(independent, dependent, '.')
            plt.show()

        elif self.get_shape(training_data)[1] > 2:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(independent[:, 0], independent[:, 1], dependent, color='#ef1234')
            plt.show()

        else:
            raise ValueError("Data too dimensional to plot")

    '''
    Calculares the Root Mean Squared Error of the model
    
    :params {Numpy.Array} y, y_hat
    :return {int} rmse
    '''
    def calculate_rmse(self, y, y_hat):
        rmse = np.sqrt(sum((y - y_hat) ** 2) / len(y))
        return rmse

    '''
    Calculates the R2-measure of the model
    
    :params {Numpy.Array} y, y_hat
    :return {int} r2
    '''
    def calculate_r2(self, y, y_hat):
        mean_y = np.mean(y)
        ss_tot = sum((y - mean_y) ** 2)
        ss_res = sum((y - y_hat) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    '''
    Evaluation model routine
    
    :params {Numpy.Array} y, y_hat
    :return {int} r2, rmse
    '''
    def evaluate_model(self, y, y_hat):
        r2 = self.calculate_r2(y, y_hat)
        rmse = self.calculate_rmse(y, y_hat)
        return r2, rmse

    '''
    Run routine of Regression
    
    :param {Numpy.Array} training_data
    :return {MultivariateLR or UnivariateLR} lr
    '''
    def run(self, training_data):
        x, y = self.get_data(self.get_columnNames(training_data), training_data)
        if len(self.get_columnNames(training_data)) > 2:
            lr = MultivariateLR(x, y, 0.0001, 10000)
            return lr
        lr = UnivariateLR(x, y)
        return lr


class MultivariateLR(Regression):

    def __init__(self, x, y, learning_rate, iterations):
        self.size = len(x)
        self.x0 = np.ones(self.size)
        self.x = x
        self.y = y
        self.B = np.zeros((self.x.shape[1]+1,), dtype=int)
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.x = self.prepare_x()

    '''
    Prepares x for calculations, adding one column of 1's
    
    :return {Numpy.Array} self.x
    '''
    def prepare_x(self):
        self.x = np.concatenate((np.ones(self.size)[:, np.newaxis], self.x), axis=1)
        return self.x

    '''
    Returns the cost at a given iteration
    
    :return {int} cost
    '''
    def cost_function(self):
        cost = np.sum((self.x.dot(self.B) - self.y) ** 2) / (2 * self.size)
        return cost

    '''
    Performs the gradient descent optimization algofithm
    
    :return {int} B, {list} cost_history
    '''
    def gradient_descent(self):
        cost_history = [0] * self.iterations
        
        for iteration in range(self.iterations):
            y_hat = self.x.dot(self.B)
            loss = y_hat - self.y
            gradient = self.x.T.dot(loss) / self.size
            B = self.B - self.learning_rate * gradient
            cost = self.cost_function()
            cost_history[iteration] = cost

        return B, cost_history

    '''
    Plot the cost history
    
    :params {list} cost_history
    '''
    def plot_cost(self, cost_history):
        plt.plot(self.iterations, cost_history)
        plt.show()

    '''
    Returns a numpy array with the predictions
    
    :params {Numpy.Array} B
    :return {Numpy.Array} y:hat
    '''
    def predict(self, B):
        y_hat = self.x.dot(B)
        return y_hat

    '''
    Run routine of MultivariateLR
    
    :return {Numpy.Array} B
    '''
    def run(self, **kwargs):
        B, cost_history = self.gradient_descent()
        return B


class UnivariateLR(Regression):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = len(x)

    '''
    Computes the parameters using OLS
    
    :return {int} m, {int} b
    '''
    def get_params(self):

        first = self.size*np.sum(self.x*self.y) - (np.sum(self.y)*np.sum(self.x))
        second = self.size*np.sum(self.x**2) - (np.sum(self.x))**2

        m = first/second
        b = (np.sum(self.y)-m*np.sum(self.x))/self.size
        return m, b

    '''
    Computes the residual average error
    
    :param {int} m, {int} b
    :return {numpy.float64} residuals'''
    def get_residuals(self, m, b):
        residuals = np.sum((self.y-self.x*m-b)**2)
        return residuals

    '''
    Returns the predictions using the parameters computed above
    
    :param {int} m, {int} b
    :return {Numpy.Array}
    '''
    def fit_LR(self, m, b):
        return np.array(self.x*m + b)

    '''
    Plots the refression line and the data points
    
    :param {int} m, {int} b
    '''
    def plot_line_train(self, m, b):

        results = self.fit_LR(m, b)
        plt.scatter(self.x, self.y, color="b", s=2)
        plt.plot(self.x, results, '--', color="r")
        plt.show()

    '''
    Returns a numpy array with the predictions
    
    :param {int} m, {int} b
    :return {Numpy.Array} y_hat
    '''
    def predict(self, m, b):
        y_hat = self.x*m + b
        return y_hat

    '''
    Run routine of UnivariateLR
    
    :return {int} m,b
    '''
    def run(self, **kwargs):

        m, b = self.get_params()
        return m, b
        #self.plot_line_train(m, b)

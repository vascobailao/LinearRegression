import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def load_data(train_dir, test_dir):

    train_df = pd.read_csv(train_dir, delimiter=",")

    test_df = pd.read_csv(test_dir, delimiter=",")

    return train_df, test_df

def return_data(train_df):

    x = np.array(train_df["x"].as_matrix())
    y = np.array(train_df["y"].as_matrix())

    size = np.size(x)

    print(size)

    return x, y, size

def get_params(x, y, size):

    first = size*np.sum(x*y) - (np.sum(y)*np.sum(x))

    second = size*np.sum(x**2) - (np.sum(x))**2

    print("first", first)
    print("second", second)


    m = first/second
    print(m)

    b = (np.sum(y)-m*np.sum(x))/size
    print(b)

    return m, b

def get_residuals(x, y, m, b):

    return np.sum((y-x*m-b)**2)

def calculate_MRSE(x, y, m, b):
    rmse = 0

    for i in range(len(x)):
        y_pred = m*x[i] + b
        rmse += (y[i] - y_pred) ** 2

    rmse = np.sqrt(rmse / len(x))
    print("RSME")
    print(rmse)

def calculate_r2(x, y, m, b):
    ss_t = 0
    ss_r = 0
    mean_y = np.mean(y)
    for i in range(len(x)):
        y_pred = b + m * x[i]
        ss_t += (y[i] - mean_y) ** 2
        ss_r += (y[i] - y_pred) ** 2
    r2 = 1 - (ss_r / ss_t)
    print("R2")
    print(r2)



def plot_data(x, y):
    plt.plot(x, y)
    plt.show()

def fit_LR(x, m, b):
    y = m*x+b
    print(y)
    return y

def plot_line_train(x, y, m, b):

    results = np.array(fit_LR(x, m, b))
    plt.scatter(x, y, color="b", s=2)
    plt.plot(x, results, '--', color="r")
    plt.show()


def plot_line_test(x_test, y_test, m, b):

    f = lambda x: m * x + b
    plt.scatter(x, y)
    plt.plot(x_test, f(x_test), 'r--')
    plt.show()

def plot_residuals(points):
    plt.plot(points)
    plt.show()


train_df, test_df = load_data(train_dir="/Users/vascofernandes/Desktop/teste-LR/train.csv", test_dir="/Users/vascofernandes/Desktop/teste-LR/test.csv")



x, y, size = return_data(train_df)
print(size)

x_test, y_test, size1 = return_data(test_df)

m, b = get_params(x, y, size)

calculate_MRSE(x, y, m, b)

calculate_r2(x, y, m, b)

fit_LR(x, m, b)

plot_line_train(x, y, m, b)




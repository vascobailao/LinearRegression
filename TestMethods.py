import unittest
import Model, Regression
import pandas as pd
import numpy as np

class TestMethods(unittest.TestCase):


    def setUp(self):

        self.model1 = Model.Model("./data/simple", ".csv")
        self.files = self.model1.set_files_in_directory()
        self.model2 = Model.Model("./data/student", ".csv")
        self.model3 = Model.Model("./datfff", ".csv")
        self.model4 = Model.Model("./data/student", ".txt")
        self.model1out = self.model1.set_dataframes(self.files)
        self.model2out = self.model2.set_dataframes(self.files)
        self.regression1 = Regression.Regression(self.model1out)
        self.regression2 = Regression.Regression(self.model2)


    ## Model class

    def test_check_no_directory(self):
        self.assertRaises(ValueError, self.model3.set_files_in_directory)

    def test_check_directory(self):
        self.assertEqual(type(self.model1.set_files_in_directory()), list)

    def test_extension(self):
        self.assertRaises(ValueError, self.model4.check_ext, self.model4.ext)

    def test_get_train_sets(self):
        output = self.model1.get_sets("train", self.files)
        self.assertIsInstance(output, pd.DataFrame)

    def test_get_test_sets(self):
        output = self.model1.get_sets("test", self.files)
        self.assertIsInstance(output, pd.DataFrame)

    def test_set_dataframes(self):
        output = self.model1.set_dataframes(self.files)
        self.assertIsInstance(output, dict)


    ## Regression class

    def test_split_data1(self):
        output = self.regression1.split_data()
        self.assertIsInstance(output[0], pd.DataFrame)

    def test_split_data2(self):
        output = self.regression1.split_data()
        self.assertIsInstance(output[1], pd.DataFrame)

    def test_get_size(self):
        tr, ts = self.regression1.split_data()
        output = self.regression1.get_size(tr)
        self.assertIsInstance(output, np.int64)

    def test_get_shape(self):
        tr, ts = self.regression1.split_data()
        output = self.regression1.get_shape(tr)
        self.assertIsInstance(output, tuple)

    def test_get_column_names(self):
        tr, ts = self.regression1.split_data()
        output = self.regression1.get_columnNames(tr)
        self.assertIsInstance(output, list)

    def test_get_training_data(self):
        output = self.regression1.get_trainingData()
        self.assertIsInstance(output, pd.DataFrame)

    def test_get_test_data(self):
        output = self.regression1.get_testData()
        self.assertIsInstance(output, pd.DataFrame)

    def test_get_data(self):
        tr, ts = self.regression1.split_data()
        cols = self.regression1.get_columnNames(tr)
        output = self.regression1.get_data(cols, tr)
        self.assertIsInstance(output, tuple)

    def test_get_data1(self):
        tr, ts = self.regression1.split_data()
        cols = self.regression1.get_columnNames(tr)
        output = self.regression1.get_data(cols, tr)
        self.assertIsInstance(output[0], np.ndarray)

    def test_get_data2(self):
        tr, ts = self.regression1.split_data()
        cols = self.regression1.get_columnNames(tr)
        output = self.regression1.get_data(cols, tr)
        self.assertIsInstance(output[1], np.ndarray)


def main():

    unittest.main()

if __name__ == '__main__':
    main()

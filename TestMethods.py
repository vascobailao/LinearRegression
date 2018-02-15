import unittest
import Model, Regression, UnivariateRegression, MultivariateRegression
import pandas as pd
from pandas.util.testing import check_output

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


    def test_split_data1(self):
        output = self.regression1.split_data()
        self.assertIsInstance(output[0], pd.DataFrame)

    def test_split_data2(self):
        output = self.regression1.split_data()
        self.assertIsInstance(output[1], pd.DataFrame)






def main():

    unittest.main()

if __name__ == '__main__':
    main()
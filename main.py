from LinearRegression import Model, Regression


def main():

    new_model = Model.Model("./data/simple", ".csv")

    files = new_model.set_files_in_directory()

    dic = new_model.set_dataframes(files)

    new_reg = Regression.Regression(dic)

    training_data, test_data = new_reg.split_data()

    column_names = new_reg.get_columnNames(training_data)

    df, df1 = new_reg.get_data(column_names, training_data)



main()

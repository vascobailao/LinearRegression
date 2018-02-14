from LinearRegression import Model, Regression


def main():

    new_model = Model.Model("./data/simple", ".csv")

    files = new_model.set_files_in_directory()

    dic = new_model.set_dataframes(files)

    new_reg = Regression.Regression(dic)

    training_data = new_reg.split_data()

    lr = new_reg.run(training_data)

    lr.run()

if __name__ == "__main__":
    main()

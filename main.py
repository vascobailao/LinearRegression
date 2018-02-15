import Model
import Regression as rg


def main():

    new_model = Model.Model("./data/student", ".csv")

    files = new_model.set_files_in_directory()

    dic = new_model.set_dataframes(files)

    new_reg = rg.Regression(dic)

    training_data = new_reg.split_data()[0]

    lr = new_reg.run(training_data)

    print(lr.run())

if __name__ == "__main__":
    main()

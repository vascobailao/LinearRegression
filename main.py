import Model
import Regression as rg


def main():

    new_model = Model.Model("./data/student", ".csv")

    files = new_model.set_files_in_directory()

    dic = new_model.set_dataframes(files)

    new_reg = rg.Regression(dic)

    training_data = new_reg.split_data()[0]

    column_names = new_reg.get_columnNames(training_data)

    print(len(column_names))

    print(new_reg.get_size(training_data))

    print(new_reg.get_shape(training_data))

    a = new_reg.get_data(columns_names=column_names, training_data=training_data)
    print("aaa")
    print(type(a))
    lr = new_reg.run(training_data)

    if lr.__class__.__name__ == "UnivariateLR":
        m, b = lr.run()
        y_hat = lr.predict(m, b)
        lr.evaluate_model(ind, y_hat)
    elif lr.__class__.__name__ == "MultivariateLR":
        B = lr.run()
        y_hat = lr.predict(B)
        lr.evaluate_model(ind, y_hat)


if __name__ == "__main__":
    main()

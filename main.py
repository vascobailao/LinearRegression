import Model
import Regression as rg

def main():

    new_model = Model.Model("./data/student", ".csv")

    files = new_model.set_files_in_directory()

    dic = new_model.set_dataframes(files)

    new_reg = rg.Regression(dic)

    training_data = new_reg.split_data()[0]

    column_names = new_reg.get_columnNames(training_data)

    ind, dep = new_reg.get_data(columns_names=column_names, training_data=training_data)

    lr = new_reg.run(training_data)

    if lr.__class__.__name__ == "UnivariateLR":
        m, b = lr.run()
        print(m, b)
        y_hat = lr.predict(m, b)
        print(lr.evaluate_model(ind, y_hat))
    elif lr.__class__.__name__ == "MultivariateLR":
        B, cost_history = lr.run()
        y_hat = lr.predict(B)
        print(lr.evaluate_model(dep, y_hat))
        lr.plot_cost(cost_history)


if __name__ == "__main__":
    main()

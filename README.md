# Linear Regression Model

Univariate and multivariate linear regression model using OLS and gradient descent optimizer

## Description

This code check a predefined directory and automatically decides wether to create a univariate regression model or a multi variate regression model.

This project is divided in 3 files:

* ``` Model.py ``` - check directory, data pre-processing
* ``` Regression.py ``` - Analyses the data and decides which sub type of regression should call
* ``` main.py ``` - Main file


## Getting Started

To run the project:

```
python main.py
```

### Prerequisites

Built with

* Python 3

Packages used:

* Numpy
* Pandas
* os
* ntpath
* matplotlib
* unittest

## Running the tests

```
python TestMethods.py
```

## Built With

* PyCharm 2017.03

## Future Work

* Folder watching - Using the "watchdog" package, check periodically for new files in the data directory
* Extend to other models (classification, etc)
* Extend to other linear regression models

## Authors

* **Vasco Fernandes**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

# coding: utf-8

__Date__ = '02 February 2019'
__author__ = 'Niraj Dev Pandey'

import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def library_check():
    if np.__version__ != '1.15.4':
        print("The project is developed on NumPy 1.15.4")
        print("you are running on numpy {} version".format(np.__version__))
    if pd.__version__ != '0.23.4':
        print("The project is developed on Pandas 0.23.4")
        print("you are running on Panda {} version".format(pd.__version__))
    if sklearn.__version__ != '0.19.2':
        print("The project is developed on Sklearn 0.19.2")
        print("you are running on sklearn {} version".format(sklearn.__version__))
    else:
        print("congratulations...! you already have all the correct dependencies installed")


library_check()


def rename_columns(DataFrame):
    """
    Change columns names in the data-set for more clearity
    :param DataFrame: data-set
    :return: changed columns name of the given data-set
    """
    DataFrame.rename(columns={'instant': 'id',
                              'dteday': 'datetime',
                              'weathersit': 'weather',
                              'hum': 'humidity',
                              'mnth': 'month',
                              'cnt': 'count',
                              'hr': 'hour',
                              'yr': 'year'}, inplace=True)


def process_datetime(datetime_columns):
    """
    Change Datetime to date and time values, set those values as columns, and
    set original datetime column as index
    :param datetime_columns: columns which needs to be converted to Pandas DateTime module
    :return: Modified columns with category having 'DateTime' which can be used for model building
    """
    datetime_columns['datetime'] = datetime_columns['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    datetime_columns['year'] = datetime_columns.datetime.apply(lambda x: x.year)
    datetime_columns['month'] = datetime_columns.datetime.apply(lambda x: x.month)
    datetime_columns['day'] = datetime_columns.datetime.apply(lambda x: x.day)
    datetime_columns['hour'] = datetime_columns.datetime.apply(lambda x: x.hour)
    datetime_columns['weekday'] = datetime_columns.datetime.apply(lambda x: x.weekday())
    datetime_columns.set_index('datetime', inplace=True)
    return datetime_columns


def drop_useless_features(DataFrame, features):
    """
    Drop specified list of features which has no impact or less in our model building
    :param DataFrame: Complete DataFrame
    :param features: Useless features which you think that needs to be dropped before fitting the model.
    :return:
    """
    DataFrame.drop(features, inplace=True, axis=1)

    if features is None:
        raise FeatureNotProvided('Please provide the list of feature which you want to drop')
    return DataFrame


def one_hot_encoding(DataFrame, categorical_features):
    """
    One hot encoder to change categorical data into dummy variable
    :param DataFrame: DataFrame
    :param categorical_features: columns which needs to be categorized as dummy
    :return: preprocessed data with dummy variable for categorical columns
    """
    DataFrame = pd.get_dummies(DataFrame, columns=categorical_features, drop_first=True)
    return DataFrame


def split_data(preprocessed_data):
    """
    Insert preprocessed data that you want to divide into train and test
    :param preprocessed_data: data ready to be split after all the cleaning and feature engineering
    :return: split data in the form of x_train. x_test, y_train & y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(
        preprocessed_data.drop('count', axis=1),
        preprocessed_data['count'], test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


def check_input_shape(x, y):
    """
   this function assume that x would be 2D input and y would be 1D array
   :param x: x data-set. it can be x_train or x_test
   :param y: y data-set. this can be y_train or y_test
   :return: error if there is any issue with the dimension of x and y
   """
    sklearn.utils.check_X_y(x, y,
                            accept_sparse=False,
                            dtype='numeric',
                            order=None,
                            copy=False,
                            force_all_finite=True,
                            ensure_2d=True,
                            allow_nd=False,
                            multi_output=False,
                            ensure_min_samples=1,
                            ensure_min_features=1,
                            y_numeric=False,
                            warn_on_dtype=False,
                            estimator=None)


def check_nan_data(x):
    """
    :param x: given any data x
    :return: Error if there is any Nan value else Pass
    """
    sklearn.utils.assert_all_finite(x)


def train_model(x_train, y_train):
    """
    :param x_train: training data x
    :param y_train: training data y
    :return: RandomForestRegressor fir pickle saved model
    """

    reg = RandomForestRegressor()
    reg.fit(x_train, y_train)
    filename = 'forest_model.sav'
    pickle.dump(reg, open(filename, 'wb'))


def test_model(saved_model, x_test):
    """
    :param saved_model: saved pickle model path
    :param x_test: x_test data
    :return: predict result on test set"""
    loaded_model = pickle.load(open(saved_model, 'rb'))
    try:
        loaded_model
    except Exception:
        raise TrainModelYourself("The loaded model is not found, please train your model by using 'Fit' function")

    result = loaded_model.predict(x_test)
    return result


def model_evaluation(y_predicted, y_true):
    """
    :param y_predicted: predicted y by your model
    :param y_true: True y
    :return: mean squared log error
    """
    error = mean_squared_log_error(y_predicted, y_true)
    return error


def main():
    data = pd.read_csv('./Bike-Sharing-Dataset/hour.csv')
    saved_model = 'forest_model.sav'
    rename_columns(data)
    data = drop_useless_features(data, ['casual', 'atemp'])
    data = process_datetime(data)
    data = one_hot_encoding(data, ['season',
                                   'holiday',
                                   'workingday',
                                   'weather',
                                   'month',
                                   'hour',
                                   'weekday'])

    x_train, x_test, y_train, y_test = split_data(data)
    check_nan_data(x_train)
    check_nan_data(y_train)
    check_nan_data(x_test)
    check_nan_data(y_test)
    check_input_shape(x_train, y_train)
    check_input_shape(x_test, y_test)
    train = train_model(x_train, y_train)
    test = test_model(saved_model, x_test)
    error = model_evaluation(test, y_test)
    print('Here are top 100 prediction by our model')
    print(test[:100])
    print('The mean square log error is:', error)


if __name__ == "__main__":
    main()

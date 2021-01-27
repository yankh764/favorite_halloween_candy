import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

class PrepareData():
    """Class that explores a bit the dataset and splits it"""

    def __init__(self, saving_dir):
        self.saving_dir = saving_dir


    def load_data(self, data_path):
        """Function that load csv file into pandas frame"""
        self.data_frame = pd.read_csv(data_path)


    def pandas_info(self):
        """A function that prints info about the data frame"""
        self.data_frame.info()
        print("#"*94)
        print(self.data_frame[:10])
        print("#"*94)
        print(self.data_frame.describe())


    def split_train_test(self):
        """Function to split the data set to a train set and test set"""
        self.train_set = self.data_frame.drop(['competitorname'], axis=1)[:80][:]
        #I will make small test set due to the small size of the dataset
        self.test_set = self.data_frame.drop(['competitorname'], axis=1)[80:][:]
        joblib.dump(self.train_set, f'{self.saving_dir}train_set.pkl')
        joblib.dump(self.test_set, f'{self.saving_dir}test_set.pkl')


    def split_labels(self):
        """
        A function that will split the data set to
        target label (y) and feature lables (x)
        """
        X_train = self.train_set.drop(['winpercent'], axis=1)
        y_train = self.train_set['winpercent']
        X_test = self.test_set.drop(['winpercent'], axis=1)
        y_test = self.test_set['winpercent']
        joblib.dump(X_train, f'{self.saving_dir}X_train.pkl')
        joblib.dump(y_train, f'{self.saving_dir}y_train.pkl')
        joblib.dump(X_test, f'{self.saving_dir}X_test.pkl')
        joblib.dump(y_test, f'{self.saving_dir}y_test.pkl')


prep_data = PrepareData(saving_dir = 'pickles/data/')
prep_data.load_data(data_path='~/Projects/machine_learning/candy/candy-data.csv',)
prep_data.pandas_info()
prep_data.split_train_test()
prep_data.split_labels()

import pandas as pd
import numpy as np
import joblib

class DataPipeline():
    """
    This is the pipeline class and we will use it
    every time b4 passing any data to the algorithim
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y


    def add_new_labels(self):
        """
        This function is going to add
        all the promising combinations we find out earlier
        """
        #List for the labels name that had negative correlation with chocolate
        negative_corr = ['fruity', 'pluribus', 'hard']
        #Thats for the postive ones
        positive_corr = [None]
        chocolate_label = self.X['chocolate']
        for label_name in negative_corr:
            new_label_values = []
            label = self.X[label_name]
            for index in range(len(label)):
                if chocolate_label[index] == 1 and label[index] == 0:
                    new_label_values.append(1)
                else:
                    new_label_values.append(0)
            new_label_name = 'chocolate_&_not_' + label_name
            self.X[new_label_name] = new_label_values


    def drop_labels(self):
        """This function will drop the unuseful and not related labels"""
        #List for all the unuseful labels
        unuseful_labels = ['caramel', 'nougat', 'peanutyalmondy',
                            'hard', 'pluribus']
        self.X = self.X.drop(unuseful_labels, axis=1)


    def convert_to_numpy(self):
        """
        This function will convert the data frames to
        a numpy array and pickle them
        """
        prepared_data_dir = 'pickles/prepared_data/'
        self.X = self.X.to_numpy()
        self.y = self.y.to_numpy()


    def add_degrees(self, degrees):
        """
        The function is going to add degrees
        to the data to make it polynomial
        """
        X_poly = np.ones((len(self.X), 1))
        for degree in range(degrees):
            X_pow = np.power(self.X, degree+1)
            X_poly = np.append(X_poly, X_pow.reshape(-1, self.X.shape[1]), axis=1)
        self.X = X_poly


    def pickle(self):
        """The function will pickle and save the prepared data"""
        prepared_data_dir = 'pickles/prepared_data/'
        X_name = input("Enter the new X pickle file's name: ")
        y_name = input("Enter the new y pickle file's name: ")
        joblib.dump(self.X, f'{prepared_data_dir}{X_name}.pkl')
        joblib.dump(self.y, f'{prepared_data_dir}{y_name}.pkl')



data_dir = 'pickles/data/'
X_name = input("Enter the X pickle file's name: ")
y_name = input("Enter the y pickle file's name: ")
X = joblib.load(f'{data_dir}{X_name}.pkl')
y = joblib.load(f'{data_dir}{y_name}.pkl')

transform_data = DataPipeline(X ,y)
transform_data.add_new_labels()
transform_data.drop_labels()
transform_data.convert_to_numpy()
transform_data.add_degrees(9)
transform_data.pickle()

import matplotlib.pyplot as plt
import numpy as np
import joblib

class LinearRegression():
    """
    This class will train the data and predict it
    plus plotting the cost function
    """

    def __init__(self, learn_rate=0.1, epochs=2000000, early_stop=1.5):
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.early_stop = early_stop
        self.model_dir = 'pickles/models/'


    def fit(self, X, y):
        """A function to implement gradient descent"""
        self.X = X
        self.y = y
        m = len(self.X)
        #Best loss for the early stopping
        best_loss = np.infty
        #Random initializations to the model's parameters
        theta = np.random.randn(82, 1)
        #List for the errors in every epoch
        self.error = []
        #Epoch number list for the plotting
        self.epoch_cnt = []
        for epoch in range(self.epochs):
            self.epoch_cnt.append(epoch)
            #Make prediction
            y_pred = self.X.dot(theta)
            #Calculate prediction's error
            epoch_mse = 1/m * np.sum(np.square(self.y - y_pred))
            #Calculate error
            error = y_pred - y
            #Print info about the model every 500 epochs
            if epoch % 500 == 0:
                print(f"Epoch number:{epoch}, Loss:{epoch_mse}")
            #Calculate gradients
            grad = 2/m * self.X.T.dot(error)
            theta = theta - self.learn_rate * grad
            #Append epochs mse to the list
            self.error.append(epoch_mse)
            #Making early stopping
            if epoch_mse < self.early_stop:
                print('\n\tEarly stopping!')
                break
        ques = input("Do you want to save the theta (y/n): ")
        if ques != 'n':
            joblib.dump(theta, f'{self.model_dir}lin_reg.pkl')
            print('\nThe model has been saved.')
        else:
            pass


    def plot_error(self, fig_dir=None):
        """The function will plot epochs error"""
        plt.plot(self.epoch_cnt, self.error, 'r-')
        plt.show()
        ques = input("Do you want to save the fig (y/n): ")
        if ques != 'n':
            plt.savefig(f'{fig_dir}gradient_epochs_error', format='png')


    def predict(self, X, theta=None):
        """
        A function to predict values according to the given theta
        or the computed one from the fit function (if there was one)
        """
        pred = np.sum(X.dot(theta))
        print(pred)





data_dir = 'pickles/prepared_data/'
X = joblib.load(f'{data_dir}X_train.pkl')
y = joblib.load(f'{data_dir}y_train.pkl')

lin_reg = LinearRegression()
#lin_reg.fit(X, y)
#lin_reg.plot_error(fig_dir='figures/train_set_figs/')
theta = joblib.load('pickles/models/lin_reg.pkl')
lin_reg.predict(X[1, :], theta)





### NOT COMPLETEDDDDD ###

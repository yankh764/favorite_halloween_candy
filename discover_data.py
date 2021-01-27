import matplotlib.pyplot as plt
import pandas as pd
import joblib


class DiscData():
    """In this class im going to discover the training data and visualize it"""

    def __init__(self, data_dir, fig_dir):
        self.data_dir = data_dir
        self.fig_dir = fig_dir
        self.train_set = joblib.load(f"{self.data_dir}train_set.pkl")


    def plot_data(self):
        """Function to visualize the train set """
        X_train = joblib.load(f"{self.data_dir}X_train.pkl")
        y_train = joblib.load(f"{self.data_dir}y_train.pkl")
        plt.plot(X_train, y_train, 'b.')
        plt.xlabel("X", fontsize=18)
        plt.ylabel("Y", fontsize=18)
        plt.title('Data vs Win Percent')
        plt.savefig('figures/train_set_figs/data_visualization', format='png')
        plt.show()


    def plot_hist(self):
        """A function that plots the attribute's histogram"""
        self.train_set.hist(figsize=(30,30))
        plt.title('Attributes Histogram')
        plt.savefig(f'{self.fig_dir}attributes_hist', format='png')
        plt.show()


    def corr_matrix(self):
        """This function will plot correlatioins matrix and print them"""
        labels = []
        for name in self.train_set.columns:
            labels.append(name)
        corr = self.train_set.corr()
        plt.matshow(corr, cmap='YlOrRd')
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.colorbar()
        plt.title("Correlatioin Matrix")
        plt.savefig(f'{self.fig_dir}corr_plot', format='png')
        plt.show()
        print('Win percent and features correlatioin:')
        print(corr['winpercent'])



    def label_vs_win(self):
        """
        Since we found a good correlatioin between few labels
        and winpercent we are going to plut them againts each other
        """
        ques = input('Do you want to plot any figures (y/n): ')
        if ques == 'n':
            ques = False
        else:
            ques = True
        while ques:
            label = input("Enter label name: ")
            self.train_set.plot(x=label, y='winpercent', kind='scatter')
            plt.title(f'{label} vs Win Percent')
            plt.savefig(f'{self.fig_dir}{label}_vs_win', format='png')
            plt.show()

            ques = input('Do you want to plot another figure (y/n): ')
            if ques == 'n':
                ques = False
            else:
                continue


    def disc_chocolate_and_new_label(self):
        """
        The function is going to discover new labels that are related to
        choclate (since its the most correlated and relavent to the win percent)
        and its going to print the new correlatioin matrix
        -The correlatioins i discovered chocolate: -fruity/-pluribus/-hard/bar
        """
        print("\nHere is the chocolate and features correlations: ")
        print(self.train_set.corr()['chocolate'])
        print('\n\tNow you can make some combos with it!\n')
        while True:
            chocolate_label = self.train_set['chocolate']
            label_name = input("Enter label's name: ")
            label = self.train_set[label_name]
            new_label = []
            corr_type = input("Enter correlatioin's type (+/-): ")
            if corr_type == '+':
                #The label class we want to check
                num = 1
            elif corr_type == '-':
                num = 0
            for index in range(len(self.train_set)):
                if chocolate_label[index] == 1 and label[index] == num:
                    new_label.append(1)
                else:
                    new_label.append(0)

            chocolate = 'chocolate_&_'
            new_label_name = input(f"Enter new label's name: {chocolate}")
            new_label_name = chocolate + new_label_name
            self.train_set[new_label_name] = new_label
            print('\n')
            print(self.train_set.corr()['winpercent'])
            ques = input("\nDo you want to make another label (y/n): ")
            if ques == 'n':
                break
            else:
                continue



disc_data = DiscData(data_dir='pickles/data/', fig_dir='figures/train_set_figs/')
disc_data.plot_data()
disc_data.plot_hist()
disc_data.corr_matrix()
disc_data.label_vs_win()
disc_data.disc_chocolate_and_new_label()
disc_data.label_vs_win()

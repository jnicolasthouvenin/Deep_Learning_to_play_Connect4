
import numpy as np
import pandas as pd

class DataManager:

    def __init__(self,root_folder):
        self.root_folder = root_folder
        self.name_values = "values.csv"
        self.name_labels = "labels.csv"
        pass

    def import_x_y(self,id_dataset,size):
        name_file = self.root_folder + id_dataset
        name_file_values = name_file + self.name_values
        name_file_labels = name_file + self.name_labels

        x = pd.read_csv(name_file_values).to_numpy()[0:size,]
        y = pd.read_csv(name_file_labels).to_numpy()[0:size,]

        return x,y

    def import_x_y_coupled_dataset(self,id_dataset_win,id_dataset_lost,size):
        half_size = size//2
        
        name_file_win = self.root_folder + id_dataset_win
        name_file_lost = self.root_folder + id_dataset_lost
        name_file_values_win = name_file_win + self.name_values
        name_file_labels_win = name_file_win + self.name_labels
        name_file_values_lost = name_file_lost + self.name_values
        name_file_labels_lost = name_file_lost + self.name_labels

        x = np.vstack((pd.read_csv(name_file_values_win).to_numpy()[0:half_size,],pd.read_csv(name_file_values_lost).to_numpy()[0:half_size,]))
        y = np.vstack((pd.read_csv(name_file_labels_win).to_numpy()[0:half_size,],pd.read_csv(name_file_labels_lost).to_numpy()[0:half_size,]))

        return x,y

    def create_train_test_sets(self,x,y,lenTest):
        """Shuffle the given dataset and split it into a training set and a test set"""
        
        nbInd = x.shape[0]
        shuffler = np.random.permutation(nbInd)
        x_train = x[shuffler][0:(nbInd-lenTest),]
        y_train = y[shuffler][0:(nbInd-lenTest),]

        x_test = x[shuffler][(nbInd-lenTest):nbInd,]
        y_test = y[shuffler][(nbInd-lenTest):nbInd,]

        return x_train,y_train,x_test,y_test

DATA_MANAGER = DataManager("data/")
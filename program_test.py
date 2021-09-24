
from numpy.core.arrayprint import BoolFormat
from game import *
from encoder import *
from arena import *
from dataManager import *
from network import *

class Program:

    def __init__(self,the_game):
        self.the_game = the_game
        self.best_network = readNeuralNetwork("networks/best_network")
        self.new_network = NeuralNetwork([264,30,30,30,30,1],0.6514)

    def select_move(self):
        return ARENA.select_move_net(self.best_network, self.the_game)

    def train_network(self, it = 1, EPOCHS = 1000, both_datasets = True, coupled_dataset = False, dataset = "classic", write = True):
        lenTest = 0
        if not(both_datasets):
            if coupled_dataset:
                if dataset == "classic":
                    x,y = DATA_MANAGER.import_x_y_coupled_dataset("win_","loss_",720000)
                else:
                    x,y = DATA_MANAGER.import_x_y_coupled_dataset("win_filtered_","loss_filtered_",720000)
                lenTest = 72000
            else:
                if dataset == "classic":
                    x,y = DATA_MANAGER.import_x_y("unfinished_",135000)
                else:
                    x,y = DATA_MANAGER.import_x_y("unfinished_filtered_",135000)
                lenTest = 13500
        else:
            if dataset == "classic":
                x,y = DATA_MANAGER.import_x_y_coupled_dataset("win_","loss_",720000)
                x_2,y_2 = DATA_MANAGER.import_x_y("unfinished_",135000)
                x = np.vstack((x,x_2))
                y = np.vstack((y,y_2))
            else:
                x,y = DATA_MANAGER.import_x_y_coupled_dataset("win_filtered_","loss_filtered_",720000)
                x_2,y_2 = DATA_MANAGER.import_x_y("unfinished_filtered_",135000)
                x = np.vstack((x,x_2))
                y = np.vstack((y,y_2))
            lenTest = 72000 + 13500
        x_train,y_train,x_test,y_test = DATA_MANAGER.create_train_test_sets(x,y,lenTest)

        self.new_network.supervised_learning(x_train,y_train,x_test,y_test,lenTest,it=it,EPOCH=EPOCHS,batch_size=100,dataset=dataset,write=write)

    def set_network_structure(self,sizes,learning_rate):
        self.new_network = NeuralNetwork(sizes,learning_rate)
    
    def study_against_random(self, dataset = 1, classic = False):

        if dataset == 1 and not(classic):
            net = readNeuralNetwork("networks/net_dataset_1_filters")
            score = ARENA.games_net_VS_random(net,game,nb_games=1000)[0]
        elif dataset == 2 and not(classic):
            net = readNeuralNetwork("networks/net_dataset_2_filters")
            score = ARENA.games_net_VS_random(net,game,nb_games=1000)[0]
        elif dataset == 3 and not(classic):
            net = readNeuralNetwork("networks/net_dataset_1&2_filters")
            score = ARENA.games_net_VS_random(net,game,nb_games=1000)[0]
        
        return score
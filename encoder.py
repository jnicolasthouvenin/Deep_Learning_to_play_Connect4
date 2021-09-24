
import numpy as np

from game import *

class Encoder:

    def __init__(self):
        pass

    def filter(self, arr, flt):
        ## Conv layer computation
        covLayer = [0 for i in range(20)]

        j = 0
        for clayer in range(20):
            step = 0
            for i in range(j, j+3):
                covLayer[clayer] += arr[i] * flt[step%3] +  arr[i+6] * flt[(step%3)+3] +  arr[i+12] * flt[(step%3)+6]
                step += 1
            if j%3==0 and j%6!=0 :
                j += 3
            else:
                j += 1
        
        ## Max pooling -> 4x5 becomes 2x4 
        poolLayer = []
        for col in range(4):
            for row in range(0,3,2):
                index = row + col * 4
                maximum = max(covLayer[index], covLayer[index+1], covLayer[index+4], covLayer[index+5])
                poolLayer.append(int(maximum))

        return(poolLayer)

    def encode_prediction(self,output_layer):
        """Returns the predicted class associated with the given output layer"""
        predicted_class = 0
        if output_layer < 0.5:
            predicted_class = 0
        else:
            predicted_class = 1
        return predicted_class

    def encode_board(self,game, isFiltered = False):
        """Return the encoding of the board that can be given to the neural network"""
        
        # create the 42 neurons for the current player
        turn = game.get_turn()
        turn_input = 0
        if turn == 2:
            turn = 0
            turn_input = np.zeros(42)
        else:
            turn_input = np.ones(42)
        
        board = np.ndarray.flatten(np.array(game.get_board()))

        # create the board for each player
        one_input = (board == 1).astype(int)
        two_input = (board == 2).astype(int)
        
        # concatenate all three inputs into one input of 126 elements
        final_input = np.hstack((one_input,two_input))
        final_input = np.hstack((final_input,turn_input))
        
        if not isFiltered:
            return final_input
        else:
            initial = final_input[:]
            
            filter_hline_top = [0,0,1,0,0,1,0,0,1]
            filter_hline_mid = [0,1,0,0,1,0,0,1,0]
            filter_hline_bottom = [1,0,0,1,0,0,1,0,0]
            filter_hline_left = [1,1,1,0,0,0,0,0,0]
            filter_hline_center = [0,0,0,1,1,1,0,0,0]
            filter_hline_right = [0,0,0,0,0,0,1,1,1]
            filter_diag_left = [1,0,0,0,1,0,0,0,1]
            filter_diag_right = [0,0,1,0,1,0,1,0,0]
            filter_plus = [0,1,0,1,0,1,0,1,0]
            filter_plus_full = [0,1,0,1,1,1,0,1,0]
            filter_cross = [1,0,1,0,1,0,1,0,1]
            
            filters = [filter_diag_left, filter_diag_right, filter_hline_bottom, filter_hline_mid, filter_hline_top, filter_hline_left, filter_hline_center, filter_hline_right, filter_cross, filter_plus, filter_plus_full]
        
            longArray = []
        
            data = initial[0:42]
            for rowsF in filters:   
                output_filter = self.filter(data, rowsF)
                longArray.append(output_filter)
        
            data = initial[42:84]
            for rowsF in filters:   
                output_filter = self.filter(data, rowsF)
                longArray.append(output_filter)
        
            for rowsF in range(88):
                longArray.append(int(initial[-1]))
        
            result = np.array([])
            for iter in range(len(longArray)):
                result = np.hstack((result, np.array(longArray[iter])))
        
            return result
            
ENCODER = Encoder()

import numpy as np

from game import *
from encoder import *

class Arena:

    def __init__(self):
        pass

    ########################### SELECT MOVE ###########################

    def select_move_net(self,net,the_game):
        """Select a move (int between 0 and 6) available to play in the given game, according to the given network"""
        values = np.zeros(7)
        for col in range(7):
            if the_game.board_at(col,5) == 0: # if the network can play in col
                # copy state of the game
                gameCopy = the_game.copy_state()
                # play the move
                gameCopy.place(col)
                input = ENCODER.encode_board(gameCopy, net.shape[0] == 264)
                # compute the score of the future position
                values[col] = net.forward(input)
            else:
                values[col] = 2
        
        # the network minimise the score of the position for the nest player
        move = np.argmin(values)
        return move

    def select_random_move(self,the_game):
        """Select a random move (int between 0 and 6) available to play in the given game"""

        # compute the number of available moves
        nb_available_moves = 0
        for col in range(7):
            if the_game.board_at(col,5) == 0: # can play in col
                nb_available_moves += 1

        # operate a random wheel to select where to play
        rand = random.uniform(0,1)
        cpt = 0
        placed = False
        move = 0
        for col in range(7):
            if the_game.board_at(col,5) == 0: # can play in col
                cpt += 1
                if cpt/nb_available_moves >= rand and not(placed):
                    move = col
                    placed = True
        # return the selected move
        return move

    ########################### PLAY A GAME ###########################

    def random_VS_random(self,the_game):
        """Returns the outcome of the match between two random players"""

        # initialization
        the_game.reset_game()
        color_net = the_game.get_turn()
        
        while the_game.get_win() is None:
            move = self.select_random_move(the_game)
            the_game.place(move)
            if the_game.get_win() is None:
                move = self.select_random_move(the_game)
                the_game.place(move)
                
        winner = the_game.get_win()
        if winner == color_net:
            result = 1
        elif winner == 0:
            result = 0
        else:
            result = -1
            
        return result

    def net_VS_random(self,net,first,the_game):
        """Returns the outcome of the match between a network and a random player"""
        the_game.reset_game()
        color_net = the_game.get_turn()
        
        while the_game.get_win() is None:
            if first:
                move = self.select_move_net(net,the_game)
            else:
                move = self.select_random_move(the_game)
            the_game.place(move)
            if the_game.get_win() is None:
                if not(first):
                    move = self.select_move_net(net,the_game)
                else:
                    move = self.select_random_move(the_game)
                the_game.place(move)
                
        winner = the_game.get_win()
        if winner == color_net:
            result = 1
        elif winner == 0:
            result = 0
        else:
            result = -1
            
        return result

    def net_VS_net(self,net_1,net_2,the_game):
        game.reset_game()
        color_net_1 = game.get_turn()
        
        while game.get_win() is None:
            move = self.select_move_net(net_1,the_game)
            the_game.place(move)
            if game.get_win() is None:
                move = self.select_move_net(net_2,the_game)
                the_game.place(move)
                
        winner = game.get_win()
        if winner == color_net_1:
            result = 1
        elif winner == 0:
            result = 0
        else:
            result = -1
            
        return result

    ########################### PLAY MULTIPLE GAMES ###########################

    def games_net_VS_random(self,net,the_game,nb_games=200):
        """Returns the average win, loss and null ratio of the network playing against random player"""

        nb_wins = 0
        nb_null = 0
        nb_loss = 0
        
        for it in range(nb_games):
            result = self.net_VS_random(net,True,the_game)
            if result == 1:
                nb_wins += 1
            elif result == 0:
                nb_null += 1
            else:
                nb_loss += 1

        for it in range(nb_games):
            result = self.net_VS_random(net,False,the_game)
            if result == 1:
                nb_wins += 1
            elif result == 0:
                nb_null += 1
            else:
                nb_loss += 1

        return nb_wins/nb_games,nb_null/nb_games,nb_loss/nb_games

    def games_net_VS_net(self,net_1,net_2,the_game,nb_games=400):
        """Returns the average win, loss and null ratio of one network playing against another"""

        nb_wins = 0
        nb_null = 0
        nb_loss = 0

        halfGames = nb_games//2
        
        for it in range(halfGames):
            result = self.net_VS_net(net_1,net_2,the_game)

            if result == 1:
                nb_wins += 1
            elif result == 0:
                nb_null += 1
            else:
                nb_loss += 1
        
        for it in range(halfGames):
            result = self.net_VS_net(net_2,net_1,the_game)

            if result == 1:
                nb_loss += 1
            elif result == 0:
                nb_null += 1
            else:
                nb_wins += 1

        return nb_wins/(2*halfGames),nb_null/(2*halfGames),nb_loss/(2*halfGames)

ARENA = Arena()
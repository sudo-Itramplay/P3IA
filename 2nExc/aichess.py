#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import board
import random
import numpy as np
from typing import List
from agent import Agent  as paco

from itertools import permutations

class Aichess():
    """
    A class to represent the game of chess.

    """
    def __init__(self):

        self.chess = board.Board()

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.depthMax = 8 
        # Dictionary to reconstruct the visited path
        self.dictPath = {}
        # Prepare a dictionary to control the visited state and at which
        # depth they were found for DepthFirstSearchOptimized
        self.dictVisitedStates = {}  
        self.best_reward = -float('inf')
        self.min_steps_for_best_reward = float('inf')
        self.optimal_path_key = None
        self.qtables = {}
    
    def print_optimal_path_from_qtable(self, agent):
        """
        Prints the optimal path found, based on the Q-table developed by the agent.
        """
        temp_aichess = Aichess()
        temp_aichess.depthMax = self.depthMax
            
        print(f"\nOptimal Path, Max Reward Found: {self.best_reward}, Number of Steps: {self.min_steps_for_best_reward}")
        print("--------------------------------------------------------------------------")

        temp_aichess.chess.reset_environment()
        current_state_pieces = temp_aichess.chess.getCurrentState()
        state_key = self.state_to_key(current_state_pieces)
        
        path_steps = 0
        #We try and prevent looping
        max_path_length = self.min_steps_for_best_reward + 50 if self.min_steps_for_best_reward != float('inf') else 100 

        move_sequence = []
        
        print(f"Initial State:")
        temp_aichess.chess.print_board()

        while path_steps < max_path_length:
            
            #We keep the track if the black king gets eaten, in which case we stop
            bk_state = self.getPieceState(current_state_pieces, 12)
            if bk_state is None:
                print(f"step {path_steps}: THE BLACK KING HAS BEEN CAPTURED. Ending path. BAD APPROACH")
                break

            #We seach fo the chakemate condition, the proper goal, instead of eating the black king
            is_checkmate = temp_aichess.isBlackInCheckMate(current_state_pieces)
            if is_checkmate:
                #If the agent has achieved checkmate, we stop
                print(f"step {path_steps}: CHECKMATE! (Goal State).")
                break

            #At last we check for draw conditions, stalemtes that shoudn't happen, at least not frequently.
            legal_next_states = temp_aichess.chess.get_all_next_states(current_state_pieces, True)
            
            if not legal_next_states:
                print(f"step {path_steps}: Stalemate.")
                break
            
            num_actions = len(legal_next_states)
            
            #Using the learned Q-table, we select the best action, in a greedy manner
            action_index = agent.max_Q(state_key, num_actions) 
            
            if action_index < 0 or action_index >= num_actions:
                print(f"step {path_steps + 1}: ERROR POLICY {action_index}. Stopping path.")
                break
            
            next_state_pieces = legal_next_states[action_index]
            
            #We get the movement made
            move_info = self.getMovement(current_state_pieces, next_state_pieces)
            
            #Just a safety check
            if move_info[0] is None or move_info[1] is None:
                print(f"step {path_steps + 1}: ERROR IN THE MOVEMENT")
                break
            #Informations about the movement
            from_pos = move_info[0][0:2]
            to_pos = move_info[1][0:2]
            piece_id = move_info[0][2]

            #store the movement in the sequence
            move_sequence.append(f"{piece_id}:{from_pos} -> {to_pos}")

            #We actually move the piece in the temp board
            temp_aichess.chess.movePiece(current_state_pieces, next_state_pieces)

            print(f"step {path_steps + 1}: Move from {from_pos} to {to_pos}")
            temp_aichess.chess.print_board()
            
            current_state_pieces = next_state_pieces
            state_key = self.state_to_key(current_state_pieces)
            path_steps += 1

        #This condition should only appear in case there is a loop in the path, or the chekcmate is never reached
        if path_steps == max_path_length:
            print(f"ERROR {max_path_length} steps (max length reached).")

        print("\nmovement sequence")
        print(" -> ".join(move_sequence))
        
    def copyState(self, state):
        #This function just pretends to copy a state
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState


    def isSameState(self, a, b):
        #This function checks if two states are the same, regardless of the order of pieces in the list
        isSameState1 = True
        for k in range(len(a)):
            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):
            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

    def getWhiteState(self, current_state):
        return self.chess.getWhiteState(current_state)

    def getBlackState(self, current_state):
        return self.chess.getBlackState(current_state)
  
    def getNextPositions(self, state):
        #Wrapper for board method, all the board related matters are administrated from there
        return self.chess.getNextPositions(state)
    
    def getListNextStatesW(self, myState, rivalState=None):
        return self.chess.getListNextStatesW(myState, rivalState)

    def getListNextStatesB(self, myState, rivalState=None):
        return self.chess.getListNextStatesB(myState)
    
    def get_all_next_states(self, current_state, color):
        return self.chess.get_all_next_states(current_state, color)

    def isVisited(self, mystate):
        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))
            isVisited = False
            for j in range(len(perm_state)):
                for k in range(len(self.listVisitedStates)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True
            return isVisited
        else:
            return False 

    def state_to_key(self, state):
        #this function converts a state (list of pieces with positions) to a tuple of tuples
        sorted_state = sorted([tuple(p) for p in state], key=lambda x: (x[2], x[0], x[1]))
        return tuple(sorted_state)


    def qLearningChess(self, agent, num_episodes, max_steps_per_episode, reward_func='simple', stochasticity=0.0):
        #This function is the core of the algorithm that trains the agent using Q-learning

        initial_state = self.chess.getCurrentState() 
        initial_state_key = self.state_to_key(initial_state)
        #We define the 4 Q-Tables we are gonna print, the first and last one, and the ones at 25% and 75% of the training
        Q_EPISODES = {0, int(num_episodes * 0.25), int(num_episodes * 0.75), num_episodes - 1}

        for episode in range(num_episodes):
            #This part is just to capture the Qtables.
            if episode in Q_EPISODES:
                q_name = str(episode)
                if episode == num_episodes - 1:
                    q_name = "100"
                elif episode == int(num_episodes * 0.25):
                    q_name = "25"
                elif episode == int(num_episodes * 0.75):
                    q_name = "75"
                    #so the names of this qtables are 0, 25, 75, 100
                #Q-table is a dict, so we deep copy it to save the snapshot
                self.qtables[q_name] = copy.deepcopy(agent.getQtable())            

            self.chess.reset_environment()
            #we get the initial state at the beginning of each episode
            current_state = self.chess.getCurrentState()
            self.listVisitedStates = []
            done = False
            total_reward = 0
            
            state_key = self.state_to_key(current_state)
            #We print the episode number and reset the total reward, to keep a debug track
            print(f" start ")
            print(f" Episode {episode+1} (Reward: {total_reward}) ---")

            for step in range(max_steps_per_episode):
                legal_next_states = self.chess.get_all_next_states(current_state, True)
                
                if not legal_next_states:
                    #stalemate is ot wanted, great penalty
                    reward = -500
                    done = True
                    break
                
                num_actions = len(legal_next_states)
                #We choose an action using the current state key
                if episode != (num_episodes-1):
                    action_index = agent.policy(state_key, num_actions) 
                else:
                    action_index = agent.policy_last_iteration(state_key, num_actions) 

                #This implementation of the drunken sailor will only be activitated with the stochasticity parameter greater tha 0.0
                if stochasticity > 0.0 and random.random() < stochasticity:
                    #Get all legal action indices
                    all_actions = list(range(num_actions))
                    #Create a list of all legal actions EXCEPT the one chosen by the policy
                    alternative_actions = [i for i in all_actions if i != action_index]
                    
                    if alternative_actions:
                        #The agent 'stumbles' and chooses a random alternative action
                        action_index = random.choice(alternative_actions)

                final_next_state = legal_next_states[action_index]
                
                #We need to keep track of the black king state, if it's disapeard there may be a proeblem
                bk_state = self.getPieceState(final_next_state, 12)
                
                reward = 0
                is_checkmate = False
                is_draw = False

                if bk_state is None:
                    #We don't want the agent to kill the bk but to chekcmate it, so we penalize heavily
                    reward = -500
                    done = True
                else:
                    #In the case the king hasn't died, we check for checkmate and draw conditions
                    is_checkmate = self.isBlackInCheckMate(final_next_state)
                    is_draw = self.is_Draw(final_next_state)
                    #Depending on the reward function selected, we calculate the reward
                    if reward_func == 'heuristic':
                        reward = self.heuristica(final_next_state, step)

                    elif reward_func == 'simple':
                        reward = -1

                    if is_checkmate:
                        #In case of checkmate, we give a high reward, and there is the end of the episode
                        reward += 100
                        done = True                   

                    if is_draw:
                        reward = -50
                        done = True 
                   
                
                total_reward += reward
                
                next_state_key = self.state_to_key(final_next_state)
                
                #make the agent learn from the transition
                agent.learn(state_key, action_index, reward, next_state_key, done, num_actions)
                
                #We actually move the piece in the board
                self.chess.movePiece(current_state, final_next_state)
                
                current_state = final_next_state
                state_key = next_state_key

                if done:
                    break

            #search for the optimal path found so far
            if total_reward > self.best_reward or (total_reward == self.best_reward and step + 1 < self.min_steps_for_best_reward):
                
                self.best_reward = total_reward
                self.min_steps_for_best_reward = step + 1
                self.optimal_path_key = initial_state_key

            print(f"final")
            print(f"Episode {episode+1} (Reward: {total_reward})")
            self.chess.print_board()
            
            if episode % 200 == 0: 
                agent.reduce_exploration_rate_by_decrease_rate() 

            if episode % 1000 == 0: 
                agent.reduce_exploration_rate_by_30_percent() 
        
        print("\n" + "="*80)
        print("  final path found  ")
        print("="*80)
        self.print_optimal_path_from_qtable(agent)
        
        return agent.q_table

    def getMovement(self, state, nextState):
        #Given a state and a successor state, return the postiion of the piece that has been moved in both states
        pieceState = None
        pieceNextState = None
        for piece in state:
            if piece not in nextState:
                movedPiece = piece[2]
                pieceNext = self.getPieceState(nextState, movedPiece)
                if pieceNext != None:
                    pieceState = piece
                    pieceNextState = pieceNext
                    break

        return [pieceState, pieceNextState]
        
    def mean(self, values):
        # Calculate the arithmetic mean (average) of a list of numeric values.
        total = 0
        n = len(values)
        
        for i in range(n):
            total += values[i]

        return total / n

#TODO: revisar apartir d'aqui
    def movePieces(self, start, depthStart, to, depthTo):
        
        # To move from one state to the next we will need to find
        # the state in common, and then move until the node 'to'
        moveList = []
        # We want that the depths are equal to find a common ancestor
        nodeTo = to
        nodeStart = start
        # if the depth of the node To is larger than that of start, 
        # we pick the ancesters of the node until being at the same
        # depth
        while(depthTo > depthStart):
            moveList.insert(0,to)
            nodeTo = self.dictPath[str(nodeTo)][0]
            depthTo-=1
        # Analogous to the previous case, but we trace back the ancestors
        #until the node 'start'
        while(depthStart > depthTo):
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # We move the piece the the parerent state of nodeStart
            self.changeState(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1

        moveList.insert(0,nodeTo)
        # We seek for common node
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # Move the piece the the parerent state of nodeStart
            self.changeState(nodeStart,ancestreStart)
            # pick the parent of nodeTo
            nodeTo = self.dictPath[str(nodeTo)][0]
            # store in the list
            moveList.insert(0,nodeTo)
            nodeStart = ancestreStart
        # Move the pieces from the node in common
        # until the node 'to'
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.changeState(moveList[i],moveList[i+1])
   
    """
    -----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    ------------------------------------ <BK check> -----------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    -----------------------------------------------------------------------------------------------------------
    """   

    def isBlackInCheckMate(self, currentState):
        # Delegate check and mate detection to the superior board functions.
        # Check: Is the Black King in check?
        if self.chess.isWatchedBk(currentState):
            # Mate: Does Black (color=False) have *any* legal moves left?
            black_legal_moves = self.chess.get_all_next_states(currentState, False)
            
            # If Black is in check and has no legal moves, it's checkmate.
            if not black_legal_moves:
                return True
        return False

    """
    -----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    ------------------------------------ <White Check> -----------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    -----------------------------------------------------------------------------------------------------------
    """   
    def isWatchedWk(self, currentState):
        # boardSim already deprecated

        wkPosition = self.getPieceState(currentState, 6)[0:2]
        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)

        # If the black king has been captured, this is not a valid configuration
        if bkState is None:
            return False

        # Check all possible moves for the black king and see if it can capture the white king
        for bkPosition in self.getNextPositions(bkState):
            if wkPosition == bkPosition:
                # White king would be in check
                return True

        if brState is not None:
            # Check all possible moves for the black rook and see if it can capture the white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False

    def allWkMovementsWatched(self, currentState):

        # boardSim already deprecated
        # In this method, we check if the white king is threatened by black pieces
        # Get the current state of the white king
        wkState = self.getPieceState(currentState, 6)
        allWatched = False

        # If the white king is on the edge of the board, it may be more vulnerable
        if wkState[0] == 0 or wkState[0] == 7 or wkState[1] == 0 or wkState[1] == 7:
            # Get the state of the black pieces
            brState = self.getPieceState(currentState, 8)
            blackState = self.getBlackState(currentState)
            allWatched = True

            # Get the possible future states for the white pieces
            nextWStates = self.getListNextStatesW(self.getWhiteState(currentState))
            for state in nextWStates:
                newBlackState = blackState.copy()
                # Check if the black rook has been captured. If so, remove it from the state
                if brState is not None and brState[0:2] == state[0][0:2]:
                    newBlackState.remove(brState)
                state = state + newBlackState
                # Move the white pieces to their new state
                self.newBoardSim(state)
                # Check if the white king is not threatened in this position,
                # which implies that not all of its possible moves are under threat
                if not self.isWatchedWk(state):
                    allWatched = False
                    break

        # Restore the original board state
        self.newBoardSim(currentState)
        return allWatched


    def isWhiteInCheckMate(self, currentState):
        if self.isWatchedWk(currentState) and self.allWkMovementsWatched(currentState):
            return True
        return False
    


    """
    -----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    ------------------------------------ <Moviment value> -----------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    -----------------------------------------------------------------------------------------------------------
    """   
    def heuristica(self, state, step):
        # Mirem quin és el rook o King 
        # i els definim
        if state[0][2] == 2:
            kingPosition = state[1]
            rookPosition = state[0]
        else:
            kingPosition = state[0]
            rookPosition = state[1]

        # Example heuristic assusiming the target position for the king is (2,4).

        # Calculate the Manhattan distance for the king to reach the target configuration (2,4)
        rowDiff = abs(kingPosition[0] - 2)
        colDiff = abs(kingPosition[1] - 4)
        # The minimum of row and column differences corresponds to diagonal moves,
        # and the absolute difference corresponds to remaining straight moves
        dist = (min(rowDiff, colDiff) + abs(rowDiff - colDiff))

        if dist == 0:
            hKing = 100  # Valor alt perquè ja ha arribat
        else:
            # Multipliquem per 3 (o el pes que vulguis) per donar-li valor
            hKing = 10 * (dist)**-1


        # Calculate the Manhattan distance for the ROOK to reach the target configuration (0,0)
        rowDiff = rookPosition[0]
        colDiff = rookPosition[1]
        # The minimum of row and column differences corresponds to diagonal moves,
        # and the absolute difference corresponds to remaining straight moves
        dist = min(rowDiff, colDiff) + abs(rowDiff - colDiff)

        # Heuristic for the rook, with three different cases
        """
        here we are rewarding the rook for beeing closer to the target (0,0)

        This reward is given based on distance to (0,0) plus an extra bonus
        if it gets closer to the corner (0,0) or to the sides (0,3) or (0,5)
        1. If the rook is at (0,0), we give a high reward (3 times the inverse of the distance)
        2. If the rook is on the first row but not in the corners (0,3) or (0,5),
           we give a moderate reward (1 times the inverse of the distance)
        3. If the rook is not on the first row but is between columns 2 and 6,
           we give a small penalty (-1 times the inverse of the distance)
        4. In all other cases, we give a significant penalty (-10)
        """
        if dist == 0:
            hRook = 100

        elif rookPosition[0] == 0 and (rookPosition[1] < 3 or rookPosition[1] > 5):
            hRook = -2 *(dist)**-1

        elif rookPosition[0] < 2 and 2 <= rookPosition[1] <= 6:
            hRook = -4 * (dist)**-1

        else:
            hRook = -500

        # Total heuristic is the sum of king 3and rook heuristics
        return hKing + hRook - (10*step)


    def is_Draw(self, current_state):
        """
        Must see if there is draw in the game.
        1st condition : Insufficient Material (King vs King).
        2nd condition : 4-fold repetition of the board state.
        """

        # PRINTS COMENTATS PER FER LA IA

        if len(current_state) == 2:
            #print("There are only 2 kings.")
            return True
        
        #Instead of checking if the state has appeared before, we count how many times it has appeared, when gets to 4 stops
        repetition_count = 0
        for past_state in self.listVisitedStates:
             if self.isSameState(past_state, current_state):
                 repetition_count += 1

        #If the state has already appeared 3 times in the history, the current move is the 4th
        if repetition_count >= 3:
            #print(f"Draw per repetició de posició: La posició actual ha aparegut {repetition_count + 1} vegades.")
            return True # Draw by 4-fold repetition

        self.listVisitedStates.append(self.copyState(current_state))
        
        return False    

"""
-----------------------------------------------------------------------------------------------------------
###########################################################################################################
###########################################################################################################
------------------------------------ <main> -----------------------------------------------------------------
###########################################################################################################
###########################################################################################################
-----------------------------------------------------------------------------------------------------------
"""    

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit(usage())

    # Initialize an empty 8x8 chess board
    
    #EXECUTIONS OF ALL THE METHODS
    print("---------------------------------------------------------------------------")
    print("##########################   MINIMAX   ####################################")
    print("---------------------------------------------------------------------------")
    print("stating AI chess... ")
    aichess = Aichess()
    print("printing board")
    aichess.chess.print_board()
    paco007 = paco(learning_rate=0.7, future_weight=0.99, exploration_rate=0.9)
    aichess.qLearningChess(agent=paco007, num_episodes=2000, max_steps_per_episode=200, reward_func='simple', stochasticity=0.0)

    paco007.save_qtable_to_json("trained_agent_qtable_AICHESS.json")
    print("Q-table saved to 'trained_agent_qtable.json'")   

    
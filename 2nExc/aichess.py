#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math
import board
import time
import numpy as np
from typing import List
from agent import Agent  as paco

RawStateType = List[List[List[int]]]

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
    
    # In file: 2nExc/aichess.py

    def print_optimal_path_from_qtable(self, agent):
        """
        Reconstructs and prints the optimal path based on the final Q-table, 
        starting from the initial state of the board.
        """
        print(f"\nOptimal Path (Greedy Policy from Q-Table, Max Reward Found: {self.best_reward})")
        print("--------------------------------------------------------------------------")

        # Set up a temporary board to simulate the optimal path
        self.chess.reset_environment()
        current_state_pieces = self.chess.getCurrentState()
        state_key = self.state_to_key(current_state_pieces)
        
        path_steps = 0
        # Safety break to prevent infinite loops in non-terminal states
        max_path_length = self.min_steps_for_best_reward + 50 if self.min_steps_for_best_reward != float('inf') else 100 
        
        print(f"STARTING POSITION (Initial State):")
        self.chess.print_board()

        while path_steps < max_path_length:
            
            # 1. Check for termination condition
            # Check if Black King (ID 12) is captured
            bk_state = self.getPieceState(current_state_pieces, 12)
            if bk_state is None:
                print(f"STEP {path_steps + 1}: Checkmate! (Goal Reached)")
                break

            # 2. Get all legal next states from the current position
            # Color is True (White's turn) as Q-Learning agent always moves white
            legal_next_states = self.chess.get_all_next_states(current_state_pieces, True)
            
            if not legal_next_states:
                print(f"STEP {path_steps + 1}: Stalemate/No Legal Moves (Path Ended)")
                break
            
            num_actions = len(legal_next_states)
            
            # 3. Use the Q-table to find the best action (greedy choice)
            # max_Q is defined in agent.py and takes state_key and num_actions
            action_index = agent.max_Q(state_key, num_actions) 
            
            if action_index < 0 or action_index >= num_actions:
                print(f"STEP {path_steps + 1}: Error: Policy returned invalid action index {action_index}. Stopping path.")
                break
            
            # 4. Determine the next state based on the greedy action
            next_state_pieces = legal_next_states[action_index]
            
            # 5. Get the movement information for printing
            # self.getMovement requires access to the full piece lists
            move_info = self.getMovement(current_state_pieces, next_state_pieces)
            
            if move_info[0] is None or move_info[1] is None:
                print(f"STEP {path_steps + 1}: Error: Could not identify the moved piece. Stopping path.")
                break

            from_pos = move_info[0][0:2]
            to_pos = move_info[1][0:2]

            # 6. Apply the move to the chess board for simulation and visual printing
            # This updates the underlying self.chess.board matrix
            self.chess.movePiece(current_state_pieces, next_state_pieces)

            print(f"STEP {path_steps + 1}: Move from {from_pos} to {to_pos}")
            self.chess.print_board()
            
            # 7. Update for next iteration
            current_state_pieces = next_state_pieces
            state_key = self.state_to_key(current_state_pieces)
            path_steps += 1

        if path_steps == max_path_length:
            print(f"Path reconstruction stopped after {max_path_length} steps (max length reached).")
        
        print("--------------------------------------------------------------------------")

    def debug_episode(self, episode, current_state, reward, action_index, legal_next_states, final_next_state, is_checkmate, is_draw, step):
        """
        Debug function to print detailed information about each episode and step
        """
        print(f"\n{'='*80}")
        print(f"EPISODE {episode} - STEP {step}")
        print(f"{'='*80}")
        print(f"Total Reward so far: {reward}")
        print(f"Action Index: {action_index} / Total Legal Moves: {len(legal_next_states)}")
        print(f"Checkmate: {is_checkmate} | Draw: {is_draw}")
        print(f"\nCurrent State Pieces:")
        self.print_state_pieces(current_state)
        print(f"\nNext State Pieces:")
        self.print_state_pieces(final_next_state)
        print(f"{'='*80}\n")

    def print_state_pieces(self, state):
        """
        Print detailed information about all pieces in a state
        """
        if not state:
            print("  [EMPTY STATE]")
            return
        
        piece_names = {
            2: "White Rook",
            6: "White King",
            8: "Black Rook",
            12: "Black King"
        }
        
        for piece in state:
            row, col, piece_id = piece[0], piece[1], piece[2]
            piece_name = piece_names.get(piece_id, f"Unknown({piece_id})")
            print(f"  {piece_name}: Position ({row}, {col})")

    def print_reward_analysis(self, reward, is_checkmate, is_draw):
        """
        Analyze and explain why the reward changed
        """
        print(f"\n--- REWARD ANALYSIS ---")
        if is_checkmate:
            print(f"✓ CHECKMATE! Reward: +100")
        elif is_draw:
            print(f"○ DRAW! Reward: -30 (penalty)")
        elif reward == -1:
            print(f"✗ Regular move. Reward: -1 (step penalty)")
        else:
            print(f"? Heuristic move. Reward: {reward}")
        print(f"--- END ANALYSIS ---\n")


        """
    -----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    ------------------------------------ <Mètodes Estat> -----------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    -----------------------------------------------------------------------------------------------------------
    """
    def copyState(self, state):
        
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState


    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
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
    
    # TODO Ho necessitem?
    # FET Val ara per ara ho mantindre igual, estic veient que es fa servir en altres llocs que no pretenim modificar
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
    

    # TODO Fer crida a board pq dongui white i black
    #       Si es prefereix fer crida a board que dongui tot
    def getNextPositions(self, state):
        # Delegate to board-level implementation
        return self.chess.getNextPositions(state)
    #TODO Aquí fer un get states de black i white pot estar bé
    #   Piece sap quins moviments pot fer cada peça
    #   Board sap on està cada peça
    # Es podria gestionar aquí el next states 
    #FET, mantindrem els dos metodes, tot i aixi la logica de control dels estats estara centralitzat a una altre funio get_all_next_states
    def getListNextStatesW(self, myState, rivalState=None):
        return self.chess.getListNextStatesW(myState, rivalState)

    def getListNextStatesB(self, myState, rivalState=None):
        return self.chess.getListNextStatesB(myState)
    
    def get_all_next_states(self, current_state, color):
        return self.chess.get_all_next_states(current_state, color)


    """
    -----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    ------------------------------------ <Llogica NO AJEDREZ> -----------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    -----------------------------------------------------------------------------------------------------------
    """
    # <TODO> Unificar aquests dos o borrar un
    #FET, he decidit comentar/eliminar isVisitedSituation, centrant tot a isVisited
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
    # <\TODO> Unificar aquests dos o borrar un


    #NO SE SI REALEMTN CAL O ES FUMADA, E GEMINI M'HA DIT QUE HO FES PER EVITAR ERRORS EN LA NOVA LOGICA
    def state_to_key(self, state):
        """
        converteix la llista destats en tupla
        """
        #ordena per a garanitzar un rodre
        sorted_state = sorted([tuple(p) for p in state], key=lambda x: (x[2], x[0], x[1]))
        return tuple(sorted_state)


    # TODO Fer un enviroment .reset?
    # FET, He fet una funcio learniin, que s'haurai d'encarregar de reiniciar l'entorn i fer el loop de cada episodi
    def qLearningChess(self, agent, num_episodes, max_steps_per_episode, reward_func='simple', stochasticity=0.0):
        """
        Cicle d'entrenament de Q-Learning per als escacs.
        """
        # Estat inicial del tauler
        initial_state = self.chess.getCurrentState() 
        initial_state_key = self.state_to_key(initial_state)

        for episode in range(num_episodes):
            self.chess.reset_environment()
            # Assegurem que tenim l'estat net després del reset
            current_state = self.chess.getCurrentState()
            self.listVisitedStates = []
            done = False
            total_reward = 0
            

            state_key = self.state_to_key(current_state)

            print(f"--- (inici) ---")
            print(f"--- Episode {episode+1} (Reward: {total_reward}) ---")

            for step in range(max_steps_per_episode):
                
                legal_next_states = self.chess.get_all_next_states(current_state, True)
                
                if not legal_next_states:
                    # Stalemate (ofegat)
                    reward = -500
                    done = True
                    break
                
                num_actions = len(legal_next_states)
                
                # Triem acció usant la clau actual
                if episode != (num_episodes-1):
                    action_index = agent.policy(state_key, num_actions) 
                else:
                    action_index = agent.policy_last_iteration(state_key, num_actions) 

                final_next_state = legal_next_states[action_index]
                
                # --- CORRECCIÓ PRINCIPAL ---
                # Comprovem si el Rei Negre (ID 12) encara és al tauler
                bk_state = self.getPieceState(final_next_state, 12)
                
                reward = 0
                is_checkmate = False
                is_draw = False

                if bk_state is None:
                    # El rei ha estat capturat -> Volem checkmate, no victori
                    #VOELMQ UE APREGNUI A NO MATAR EL REI
                    reward = -500
                    done = True
                else:
                    # El rei hi és, comprovem escac i mat o taules de forma segura
                    is_checkmate = self.isBlackInCheckMate(final_next_state)
                    is_draw = self.is_Draw(final_next_state)
                    
                    if reward_func == 'heuristic':
                        reward = self.heuristica(final_next_state, step)

                    elif reward_func == 'simple':
                        reward = -2

                    if is_checkmate:
                        reward += 100
                        done = True                   

                    if is_draw:
                        reward = -50
                        done = True 
                   
                
                total_reward += reward
                
                # Calculem la clau del següent estat UNA sola vegada
                next_state_key = self.state_to_key(final_next_state)
                
                # Aprenentatge
                agent.learn(state_key, action_index, reward, next_state_key, done, num_actions)
                
                # Movem peça visualment (això actualitza self.board)
                # Important: fer-ho abans de canviar current_state
                self.chess.movePiece(current_state, final_next_state)
                
                # Actualitzem variables per la següent iteració
                current_state = final_next_state
                state_key = next_state_key

                if done:
                    break

                        
            #Search for the betst path to get to checkmate
            if total_reward > self.best_reward or \
               (total_reward == self.best_reward and step + 1 < self.min_steps_for_best_reward):
                
                self.best_reward = total_reward
                self.min_steps_for_best_reward = step + 1
                self.optimal_path_key = initial_state_key

            print(f"--- (FINAL) ---")
            print(f"--- Episode {episode+1} (Reward: {total_reward}) ---")
            self.chess.print_board()
            
            if episode % 200 == 0: 
                agent.reduce_exploration_rate_by_decrease_rate() 

            if episode % 1000 == 0: 
                agent.reduce_exploration_rate_by_30_percent() 
        
        print("\n" + "="*80)
        print("  Final PATH  ")
        print("="*80)
        self.print_optimal_path_from_qtable(agent)
                
        return agent.q_table

    def getMovement(self, state, nextState):
        # Given a state and a successor state, return the postiion of the piece that has been moved in both states
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
    
    def reconstructPath(self, state, depth):
        # Once the solution is found, reconstruct the path taken to reach it
        for i in range(depth):
            self.pathToTarget.insert(0, state)
            # For each node, retrieve its parent from dictPath
            state = self.dictPath[str(state)][0]

        # Insert the root node at the beginning
        self.pathToTarget.insert(0, state)
        
    def mean(self, values):
        # Calculate the arithmetic mean (average) of a list of numeric values.
        total = 0
        n = len(values)
        
        for i in range(n):
            total += values[i]

        return total / n


    def standard_deviation(self, values, mean_value):
        # Calculate the standard deviation of a list of values.
            total = 0
            n = len(values)

            for i in range(n):
                total += pow(values[i] - mean_value, 2)

            return pow(total / n, 1 / 2)

    """
    -----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    ------------------------------------ <Llogica Ajedrez> -----------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    -----------------------------------------------------------------------------------------------------------
    """

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
    ------------------------------------ <CODIS EXEMPLE> -----------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    -----------------------------------------------------------------------------------------------------------
    """
    def expectimax(self, depthWhite, depthBlack):
        
        currentState = self.chess.getCurrentState()
        color = True
        currentState = self.chess.getCurrentState()
        self.newBoardSim(currentState)

        #main game loop
        while (not self.is_Draw(currentState)) and (not self.isBlackInCheckMate(currentState) and not self.isWhiteInCheckMate(currentState)):
                    
            #choose depth depending on the turn
            profunditat = depthWhite if color else depthBlack

            self.initialdepth = profunditat

            #recursive call
            valor, millors_peces_mogudes = self.recursiveexpectimax(currentState, profunditat, color)

            #check if there are no more moves
            if millors_peces_mogudes is None:
                print(f"No s'ha trobat cap moviment vàlid per a {'Blanques' if color else 'Negres'}. (Possible ofegat?)")
                break
                
            #take the actual movement
            moviment = self.getMovement(self.chess.getCurrentState(), millors_peces_mogudes)
            
            if moviment[0] is None or moviment[1] is None:
                print("Error a 'getMovement', no s'ha trobat el moviment. Avortant.")
                print("Estat actual:", currentState)
                print("Peçes mogudes:", millors_peces_mogudes)
                break 
            
            #make the movement
            from_pos = moviment[0][0:2]
            to_pos = moviment[1][0:2]
            self.chess.moveSim(from_pos, to_pos)
                        
            
            print(f"Torn de {'Blanques' if color else 'Negres'} (Expectimax)")
            print(f"Movem {from_pos} a {to_pos} (Valor esperat: {valor})")
            self.chess.print_board()
            
            #update the state and change turn
            currentState = self.chess.getCurrentState()
            color = not color
        #final print
        if self.isBlackInCheckMate(currentState):
            print("Escac i mat! Guanyen les blanques.")
        elif self.isWhiteInCheckMate(currentState):
            print("Escac i mat! Guanyen les negres.")
        else:
            print("Partida acabada (sense escac i mat).")
            if self.is_Draw(currentState):
                print("La partida ha acabat en empat (taules).")
        return
    


    def minimaxVSalphabeta(self, depthWhite, depthBlack):
        
        currentState = self.chess.getCurrentState()    
        color = True
        currentState = self.chess.getCurrentState()
        self.newBoardSim(currentState)

        #main game loop
        while (not self.is_Draw(currentState)) and (not self.isBlackInCheckMate(currentState) and not self.isWhiteInCheckMate(currentState)):

            #depth depending on the turn
            profunditat = depthWhite if color else depthBlack

            if color:
                #white move minimax
                valor, millors_peces_mogudes = self.recursiveminimaxGame(currentState, profunditat, color)
            else:
                #black move alphabeta
                valor, millors_peces_mogudes = self.recursiveAlphaBetaPoda(currentState, profunditat, color, -float('inf'), float('inf'))
            #no move check
            if millors_peces_mogudes is None:
                print(f"No s'ha trobat cap moviment vàlid per a {'Blanques' if color else 'Negres'}. (Possible ofegat?)")
                break
                
            #take the actual movement
            moviment = self.getMovement(self.chess.getCurrentState(), millors_peces_mogudes)
            
            if moviment[0] is None or moviment[1] is None:
                print("Error a 'getMovement', no s'ha trobat el moviment. Avortant.")
                print("Estat actual:", currentState)
                print("Peçes mogudes:", millors_peces_mogudes)
                break 
            
            #execute the movement
            from_pos = moviment[0][0:2]
            to_pos = moviment[1][0:2]
            self.chess.moveSim(from_pos, to_pos)           

            if not color:
                print(f"Torn de {'Blanques' if color else 'Negres'} (Expecti)")
                print(f"Movem {from_pos} a {to_pos} (Valor esperat: {valor})")
                self.chess.print_board()
            else:
                print(f"Torn de {'Blanques' if color else 'Negres'} (alpha)")
                print(f"Movem {from_pos} a {to_pos} (Valor esperat: {valor})")
                self.chess.print_board()
            
            #update state and turn
            currentState = self.chess.getCurrentState()
            color = not color

        if self.isBlackInCheckMate(currentState):
            print("Escac i mat! Guanyen les blanques.")
        elif self.isWhiteInCheckMate(currentState):
            print("Escac i mat! Guanyen les negres.")
        else:
            print("Partida acabada (sense escac i mat).")
            if self.is_Draw(currentState):
                print("La partida ha acabat en empat (taules).")
        return
    

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
    #paco007 = paco(learning_rate=0.1, future_weight=0.9, exploration_rate=0.2)
    paco007 = paco(learning_rate=0.9, future_weight=0.9, exploration_rate=0.9)
    aichess.qLearningChess(agent=paco007, num_episodes=3000, max_steps_per_episode=200, reward_func='heuristic', stochasticity=0.1)

    
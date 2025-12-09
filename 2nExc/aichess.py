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
        return self.chess.getListNextStatesB(myState, rivalState)
    
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

        for episode in range(num_episodes):
            self.chess.reset_environment()
            # Assegurem que tenim l'estat net després del reset
            current_state = self.chess.getCurrentState()
            self.listVisitedStates = []
            done = False
            total_reward = 0
            
            # OPTIMITZACIÓ: Calculem la clau inicial només un cop a l'inici de l'episodi
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
                action_index = agent.policy(state_key, num_actions) 
                final_next_state = legal_next_states[action_index]
                
                # --- CORRECCIÓ PRINCIPAL ---
                # Comprovem si el Rei Negre (ID 12) encara és al tauler
                bk_state = self.getPieceState(final_next_state, 12)
                
                reward = 0
                is_checkmate = False
                is_draw = False

                if bk_state is None:
                    # El rei ha estat capturat -> VICTÒRIA
                    reward = 100
                    done = True
                else:
                    # El rei hi és, comprovem escac i mat o taules de forma segura
                    is_checkmate = self.isBlackInCheckMate(final_next_state)
                    is_draw = self.is_Draw(final_next_state)
                    
                    if reward_func == 'heuristic':
                        reward = self.heuristica(final_next_state, True)
                    elif is_draw:
                        reward = -30
                        done = True
                    elif reward_func == 'simple':
                        reward = -1 
                    elif is_checkmate:
                        reward = 100
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

            print(f"--- (FINAL) ---")
            print(f"--- Episode {episode+1} (Reward: {total_reward}) ---")
            self.chess.print_board()
            
            if episode < 5 or episode % 100 == 0: 
                time.sleep(1) 
        
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
    def isWatchedBk(self, currentState):

        # boardSim already deprecated

        bkPosition = self.getPieceState(currentState, 12)[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # If the white king has been captured, this is not a valid configuration
        if wkState is None:
            return False

        # Check all possible moves of the white king to see if it can capture the black king
        for wkPosition in self.getNextPositions(wkState):
            if bkPosition == wkPosition:
                # Black king would be in check
                return True

        if wrState is not None:
            # Check all possible moves of the white rook to see if it can capture the black king
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True

        return False

    def allBkMovementsWatched(self, currentState):
        # In this method, we check if the black king is threatened by the white pieces

        # boardSim already deprecated
        # Get the current state of the black king
        bkState = self.getPieceState(currentState, 12)
        allWatched = False

        # If the black king is on the edge of the board, all its moves might be under threat
        if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
            wrState = self.getPieceState(currentState, 2)
            whiteState = self.getWhiteState(currentState)
            allWatched = True
            # Get the future states of the black pieces
            nextBStates = self.getListNextStatesB(self.getBlackState(currentState))

            for state in nextBStates:
                newWhiteState = whiteState.copy()
                # Check if the white rook has been captured; if so, remove it from the state
                if wrState is not None and wrState[0:2] == state[0][0:2]:
                    newWhiteState.remove(wrState)
                state = state + newWhiteState
                # Move the black pieces to the new state
                # boardSim already deprecated

                # Check if in this position the black king is not threatened; 
                # if so, not all its moves are under threat
                if not self.isWatchedBk(state):
                    allWatched = False
                    break

        # Restore the original board state
        # boardSim already deprecated
        return allWatched

    def isBlackInCheckMate(self, currentState):
        if self.isWatchedBk(currentState) and self.allBkMovementsWatched(currentState):
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

    # Aquest mètode ha d'estar aquí pq és necessari saber si està o no en chackmate per donar-las
    def heuristica(self, currentState, color):
        # This method calculates the heuristic value for the current state.
        # The value is initially computed from White's perspective.
        # If the 'color' parameter indicates Black(FALSE), the final value is multiplied by -1.

        value = 0

        bkState = self.getPieceState(currentState, 12)  # Black King
        wkState = self.getPieceState(currentState, 6)   # White King
        wrState = self.getPieceState(currentState, 2)   # White Rook
        brState = self.getPieceState(currentState, 8)   # Black Rook

        filaBk, columnaBk = bkState[0], bkState[1]
        filaWk, columnaWk = wkState[0], wkState[1]

        if wrState is not None:
            filaWr, columnaWr = wrState[0], wrState[1]
        if brState is not None:
            filaBr, columnaBr = brState[0], brState[1]

        # If the black rook has been captured
        if brState is None:
            value += 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and wrState is not None:
                filaR = abs(filaBk - filaWr)
                columnaR = abs(columnaWr - columnaBk)
                value += (min(filaR, columnaR) + abs(filaR - columnaR)) / 10

            # For White: the closer our king is to the opponent’s king, the better.
            # Subtract 7 from the king-to-king distance since 7 is the maximum distance possible on the board.
            value += (7 - distReis)

            # If the black king is against a wall, prioritize pushing him into a corner (ideal for checkmate).
            if bkState[0] in (0, 7) or bkState[1] in (0, 7):
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            # Otherwise, encourage moving the black king closer to the wall.
            else:
                value += (max(abs(filaBk - 3.5), abs(columnaBk - 3.5))) * 10

        # If the white rook has been captured.
        # The logic is similar to the previous section but with reversed (negative) values.
        if wrState is None:
            value -= 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and brState is not None:
                filaR = abs(filaWk - filaBr)
                columnaR = abs(columnaBr - columnaWk)
                value -= (min(filaR, columnaR) + abs(filaR - columnaR)) / 10

            # For White: being closer to the opposing king is better.
            # Subtract 7 from the distance since that’s the maximum possible distance.
            value += (-7 + distReis)

            # If the white king is against a wall, penalize that position.
            if wkState[0] in (0, 7) or wkState[1] in (0, 7):
                value -= (abs(filaWk - 3.5) + abs(columnaWk - 3.5)) * 10
            # Otherwise, encourage the king to stay away from the wall.
            else:
                value -= (max(abs(filaWk - 3.5), abs(columnaWk - 3.5))) * 10

        # If the black king is in check, reward this state.
        if self.isWatchedBk(currentState):
            value += 20

        # If the white king is in check, penalize this state.
        if self.isWatchedWk(currentState):
            value -= 20

        # If the current player is Black, invert the heuristic value.
        if not color:
            value *= -1

        return value


    def verify_single_piece_moved(self, state_before, state_after):
        """
        Mètode per verificar que nomes es mogui una peça

        Returns: bool: True si només una peça ha canviat de lloc, False altrament.
        """
        # Mirem quantes peces del estat inicial ja no estàn al estat final
        moved_pieces_count = 0
        for piece_position in state_before:
            if piece_position not in state_after:
                moved_pieces_count += 1

        
        return moved_pieces_count == 1



    def is_legal_transition(self, current_player_state, rival_state, moves, color):
        """
        Construeix un estat futur i comprova si la transició és legal.

        Returns: bool: (es legal==True), tupla: next_node
        """

        # Verifiquem que la llista de moviments proposada ('moves') només implica
        # el moviment d'una única peça.
        if not self.verify_single_piece_moved(current_player_state, moves):
            return False, None

        # Obtenim la informació del moviment per identificar possibles captures
        move_info = self.getMovement(current_player_state, moves)
        #Comprovació de seguretat
        if move_info[0] is None or move_info[1] is None:
            return False, None 

        # Comprovació de captura
        new_pos_coords = move_info[1][0:2]
        # Eliminem la peça rival si ha estat capturada
        new_rival_state = [p for p in rival_state if p[0:2] != new_pos_coords]


        # Construïm el node (estat) complet del següent moviment
        next_node = moves + new_rival_state

        # Comprovació de legalitat: el rei, del color que mou, no pot quedar/estar en escac
        if color and self.isWatchedWk(next_node):
            return False, None  
        
        if not color and self.isWatchedBk(next_node):
            return False, None  

        # Si totes les comprovacions passen, el moviment és legal
        return True, next_node

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
    paco007 = paco(learning_rate=0.1, future_weight=0.9, exploration_rate=0.2)
    aichess.qLearningChess(agent=paco007, num_episodes=3000, max_steps_per_episode=200, reward_func='simple', stochasticity=0.1)

    
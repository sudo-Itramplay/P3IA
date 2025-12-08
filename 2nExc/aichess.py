#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math
import board
import numpy as np
import sys
import queue
import random
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():
    """
    A class to represent the game of chess.

    """

    def __init__(self, TA, myinit=True):


        self.chess = board.Board()

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.depthMax = 8
        # Dictionary to reconstruct the visited path
        self.dictPath = {}
        # Prepare a dictionary to control the visited state and at which
        # depth they were found for DepthFirstSearchOptimized
        self.dictVisitedStates = {}



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
        
    def changeState(self, start, to):
        # Determine which piece has moved from the start state to the next state
        if start[0] == to[0]:
            movedPieceStart = 1
            movedPieceTo = 1
        elif start[0] == to[1]:
            movedPieceStart = 1
            movedPieceTo = 0
        elif start[1] == to[0]:
            movedPieceStart = 0
            movedPieceTo = 1
        else:
            movedPieceStart = 0
            movedPieceTo = 0

        # Move the piece that changed
        self.chess.moveSim(start[movedPieceStart], to[movedPieceTo])  


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
    

    # TODO Fer crida a board pq dongui white i black
    #       Si es prefereix fer crida a board que dongui tot
    def getNextPositions(self, state):
        # Given a state, we check the next possible states
        # From these, we return a list with position, i.e., [row, column]
        if state == None:
            return None
        if state[2] > 6:
            nextStates = self.getListNextStatesB([state])
        else:
            nextStates = self.getListNextStatesW([state])
        nextPositions = []
        for i in nextStates:
            nextPositions.append(i[0][0:2])
        return nextPositions
    #TODO Aquí fer un get states de black i white pot estar bé
    #   Piece sap quins moviments pot fer cada peça
    #   Board sap on està cada peça
    # Es podria gestionar aquí el next states 
    #FET, mantindrem els dos metodes, tot i aixi la logica de control dels estats estara centralitzat a una altre funio get_all_next_states
    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates
    
    def get_all_next_states(self, current_state, color):
        """
        Aquesta funcio respon al todo de les dues anteriors, centralitza la logica d'obtenir els nous estats legals.
        Retorna: llista d'estats legals seguents
        """
        self.newBoardSim(current_state) #obte el board simulat actual
        '''
        current player pieces, seria la variable del jugador que mouria
        rival pieces, seria la variable del jugador rival
        next moves raw, seria la llista de nous estats possibles sense comprovar si son legals o no
        legal next states, seria la llista final d'estats legals que es retornaria
        '''
        if color: #Blanques que seria el maximitzador
            current_player_pieces = self.getWhiteState(current_state)
            rival_pieces = self.getBlackState(current_state)
            #utilitze les dues funcions ded alt per obtenir els estxats de cada color
            next_moves_raw = self.getListNextStatesW(current_player_pieces)
        else: #negres que seria el minimitzador
            '''
            Els negre no mouran mai, tot iaxi es importnat saber els seus possibles movimetns, per considerar que podrien o no fer les blanques
            '''
            current_player_pieces = self.getBlackState(current_state)
            rival_pieces = self.getWhiteState(current_state)
            next_moves_raw = self.getListNextStatesB(current_player_pieces)

        legal_next_states = []
        for next_move_pieces in next_moves_raw:
            #retorna l'estat nou del tablero amb les movimetn que serien legals
            is_legal, next_node = self.is_legal_transition(current_player_pieces,rival_pieces,next_move_pieces,color)
            if is_legal:
                legal_next_states.append(next_node)
                
        return legal_next_states


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
    '''
    def isVisitedSituation(self, color, mystate):
        
        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == \
                            self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True

            return isVisited
        else:
            return False
    '''    
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
        cicle denremanet de qlearning pels escacs
        
        APARTAT C DEMANA APLICAR DRUNKEN SAILOR ESTARA COMENTAT
        stochasticity: Probabilidad de que el movimiento falle (drunken sailor).
        """
        #estat inicial del tauler
        initial_state = self.getCurrentSimState() 

        for episode in range(num_episodes):
            current_state = initial_state
            self.newBoardSim(current_state) #reet necesari a acada principi d'episodi
            done = False
            total_reward = 0

            self.listVisitedStates = [] 

            for step in range(max_steps_per_episode):
                
                legal_next_states = self.get_all_possible_moves(current_state, True)
                
                if not legal_next_states:
                    # Empat per afogament (Stalemate)
                    reward = 0
                    done = True
                    break
                
                state_key = self.state_to_key(current_state)
                num_actions = len(legal_next_states)
                
                #La accio que triara l'agent segons la policy aplicada a agent.py
                action_index = agent.policy_for_chess(state_key, num_actions) 
                
                # El estado al que se pretendía mover
                intended_next_state = legal_next_states[action_index]
                
                final_next_state = intended_next_state
                #logica drunken sailor
                '''
                if stochasticity > 0.0 and random.random() < stochasticity:
                    other_actions = [s for i, s in enumerate(legal_next_states) if i != action_index]
                    if other_actions:
                        final_next_state = random.choice(other_actions)
                '''
                # 4. Calcular Recompensa y Finalización
                is_checkmate = self.isBlackInCheckMate(final_next_state)
                is_draw = self.is_Draw(final_next_state)
                
                reward = 0
                
                if is_checkmate:
                    reward = 100
                    done = True
                elif is_draw:
                    reward = -10#penalitzaio per empat
                    done = True
                elif reward_func == 'simple':
                    reward = -1 
                elif reward_func == 'heuristic':
                    reward = self.heuristica(final_next_state, True)
                
                total_reward += reward
                
                next_state_key = self.state_to_key(final_next_state)
                
                #Actualitzar la Q-table de l'agent
                agent.learn_for_chess(state_key, action_index, reward, next_state_key, done, num_actions)
                
                current_state = final_next_state
                self.newBoardSim(current_state) #Sincronitzar el simulador, hem de mantenir les actualiztacion a la board
            
                if done:
                    break

            if episode in [0, num_episodes // 2, num_episodes - 1]:
                 print(f"--- Episode {episode+1} (Reward: {total_reward}) ---")
                 #imprmi qtabel

        #AQUEST RETURN NS SI REALMENT FARIA FALTA PERO XDXD
        return agent.q_table

    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

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

        self.newBoardSim(currentState)

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

        self.newBoardSim(currentState)
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
                self.newBoardSim(state)

                # Check if in this position the black king is not threatened; 
                # if so, not all its moves are under threat
                if not self.isWatchedBk(state):
                    allWatched = False
                    break

        # Restore the original board state
        self.newBoardSim(currentState)
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
        self.newBoardSim(currentState)

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

        self.newBoardSim(currentState)
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

        if len(current_state) == 2:
            print("There are only 2 kings.")
            return True
        
        #Instead of checking if the state has appeared before, we count how many times it has appeared, when gets to 4 stops
        repetition_count = 0
        for past_state in self.listVisitedStates:
             if self.isSameState(past_state, current_state):
                 repetition_count += 1

        #If the state has already appeared 3 times in the history, the current move is the 4th
        if repetition_count >= 3:
            print(f"Draw per repetició de posició: La posició actual ha aparegut {repetition_count + 1} vegades.")
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
        
        currentState = self.getCurrentState()
        color = True
        currentState = self.getCurrentState()
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
            moviment = self.getMovement(self.getCurrentSimState(), millors_peces_mogudes)
            
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
            self.chess.boardSim.print_board()
            
            #update the state and change turn
            currentState = self.getCurrentSimState()
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
        
        currentState = self.getCurrentState()    
        color = True
        currentState = self.getCurrentState()
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
            moviment = self.getMovement(self.getCurrentSimState(), millors_peces_mogudes)
            
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
                self.chess.boardSim.print_board()
            else:
                print(f"Torn de {'Blanques' if color else 'Negres'} (alpha)")
                print(f"Movem {from_pos} a {to_pos} (Valor esperat: {valor})")
                self.chess.boardSim.print_board()
            
            #update state and turn
            currentState = self.getCurrentSimState()
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
    TA = np.zeros((8, 8))


    # Load initial positions of the pieces
    TA = np.zeros((8, 8))
    TA[7][0] = 2   
    TA[7][5] = 6   
    TA[0][7] = 8   
    TA[0][5] = 12  
    
    #EXECUTIONS OF ALL THE METHODS
    print("---------------------------------------------------------------------------")
    print("##########################   MINIMAX   ####################################")
    print("---------------------------------------------------------------------------")
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    print("printing board")
    aichess.chess.boardSim.print_board()
    aichess.minimaxGame(4,4)

    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math

import chess
import board
import agent
import environment as env
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game
        
    listNextStates : list
        List of next possible states for the current player.

    listVisitedStates : list
        List of all visited states during A* and other search algorithms.

    listVisitedSituations : list
        List of visited game situations (state + color) for minimax/alpha-beta pruning.

    pathToTarget : list
        Sequence of states from the initial state to the target (used by A*).

    depthMax : int
        Maximum search depth for minimax/alpha-beta searches.

    dictPath : dict
        Dictionary used to reconstruct the path in A* search.

    Methods:
    --------
    copyState(state) -> list
        Returns a deep copy of the given state.

    isVisitedSituation(color, mystate) -> bool
        Checks whether a given state with a specific color has already been visited.

    getListNextStatesW(myState) -> list
        Returns a list of possible next states for the white pieces.

    getListNextStatesB(myState) -> list
        Returns a list of possible next states for the black pieces.

    isSameState(a, b) -> bool
        Checks whether two states represent the same board configuration.

    isVisited(mystate) -> bool
        Checks if a given state has been visited in search algorithms.

    getCurrentState() -> list
        Returns the combined state of both white and black pieces.

    getNextPositions(state) -> list
        Returns a list of possible next positions for a given state.

    heuristica(currentState, color) -> int
        Calculates a heuristic value for the current state from the perspective of the given color.

    movePieces(start, depthStart, to, depthTo) -> None
        Moves all pieces along the path between two states.

    changeState(start, to) -> None
        Moves a single piece from start state to to state.

    reconstructPath(state, depth) -> None
        Reconstructs the path from initial state to the target state for A*.

    isWatchedWk(currentState) / isWatchedBk(currentState) -> bool
        Checks if the white or black king is under threat.

    allWkMovementsWatched(currentState) / allBkMovementsWatched(currentState) -> bool
        Checks if all moves of the white or black king are under threat.

    isWhiteInCheckMate(currentState) / isBlackInCheckMate(currentState) -> bool
        Determines if the white or black king is in checkmate.

    minimaxGame(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a full game using the Minimax algorithm for both white and black.

    alphaBetaPoda(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a game where both players use Minimax with Alpha-Beta Pruning.

    expectimax(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a full game where both players use the Expectimax algorithm.

    mean(values: list[float]) -> float
        Returns the arithmetic mean (average) of a list of numerical values.

    standardDeviation(values: list[float], mean_value: float) -> float
        Computes the standard deviation of a list of numerical values based on the given mean.

    calculateValue(values: list[float]) -> float
        Computes the expected value from a set of scores using soft-probabilities 
        derived from normalized values (exponential weighting). Can be useful for Expectimax.

    ismax(depthWhite: int, depthBlack: int)


    def recursiveminimaxGame(values: state, depth, color:bool)

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

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
        self.ai = agent.Agent()       # Assumint que arregles el constructor d'Agent
        self.env = env.Environment()  # Assumint que arregles el constructor d'Environment

    def copyState(self, state):
        
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState
        
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

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

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

    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

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

    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState

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

    def reconstructPath(self, state, depth):
        # Once the solution is found, reconstruct the path taken to reach it
        for i in range(depth):
            self.pathToTarget.insert(0, state)
            # For each node, retrieve its parent from dictPath
            state = self.dictPath[str(state)][0]

        # Insert the root node at the beginning
        self.pathToTarget.insert(0, state)


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


    def calculateValue(self, values):
        # Calculate a weighted expected value based on normalized probabilities. - useful for Expectimax.
        
        # Compute mean and standard deviation
        mean_value = self.mean(values)
        std_dev = self.standard_deviation(values, mean_value)

        # If all values are equal, the deviation is 0, equal probability
        if std_dev == 0:
            return values[0]

        expected_value = 0
        total_weight = 0
        n = len(values)

        for i in range(n):
            # Normalize value using z-score
            normalized_value = (values[i] - mean_value) / std_dev

            # Convert to a positive weight using e^(-x)
            positive_weight = pow(1 / math.e, normalized_value)

            # Weighted sum
            expected_value += positive_weight * values[i]
            total_weight += positive_weight

        # Final expected value (weighted average)
        return expected_value / total_weight

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


    def recursiveminimaxGame(self, node, depth, color):
        #Color -> True = white, False = Black
        
        self.newBoardSim(node)

        #This checks are made to prevent the NoneType error, that appears when a king is missing
        bkState = self.getPieceState(node, 12)
        wkState = self.getPieceState(node, 6)

        #The case the king was missing we cosnider the game has ended
        if wkState is None: #White king is gone.
            #If the current player is black, it has won and if it is white has lost
            return -float('inf') if color else float('inf'), node
            
        if bkState is None: #Black king is gone.
            #The same from befor but backwards
            return float('inf') if color else -float('inf'), node
        
        #If the limit of movemtns is reached stop the recursion
        if depth == 0:
            return self.heuristica(node, color), node
        
        if (color and self.isWhiteInCheckMate(node)) or (not color and self.isBlackInCheckMate(node)):
            #In case the current player is in checkmate, return the worst possible score depending on who whas playing
            #return-float('inf'), node
            if color:
                return -float('inf'), node
            else:
                return float('inf'), node 

        #clean the next states list
        next_states = []

        if color: #whites are moving
            current_player_state = self.getWhiteState(node)
            rival_state = self.getBlackState(node)
            next_states = self.getListNextStatesW(current_player_state)
        else: #blacks are moving
            current_player_state = self.getBlackState(node)
            rival_state = self.getWhiteState(node)
            next_states = self.getListNextStatesB(current_player_state)

        #In case no one can move
        if not next_states:
            return self.heuristica(node, color), node
       
        #Always looking from whites perspective
        millorValor = -float('inf')
        millorMove = None

        for moves in next_states:
            #illegal movements check
            is_legal, next_node = self.is_legal_transition(current_player_state, rival_state, moves, color)

            #if the movement is illegal continue
            if not is_legal:
                continue 

            #Next turn
            self.newBoardSim(next_node)

            valor_oponent, _ = self.recursiveminimaxGame(next_node, depth - 1, not color)
            valor_actual = -valor_oponent
            if valor_actual > millorValor:
                millorValor = valor_actual
                millorMove = self.copyState(moves)

        if millorMove is None and next_states:
            millorMove = next_states[0]

        #Restore the state
        self.newBoardSim(node)

        return millorValor, millorMove
    
    def getCurrentSimState(self):
        listStates = []
        #To update the actual table state
        for i in self.chess.boardSim.currentStateW:
            listStates.append(i)
        for j in self.chess.boardSim.currentStateB:
            listStates.append(j)
        return listStates

    def minimaxGame(self, depthWhite, depthBlack):
        #In this method we simulate a full game using the Minimax algorithm for both players.
        color = True  #White starts
        currentState = self.getCurrentState()
        self.newBoardSim(currentState)
        
        while (not self.is_Draw(currentState)) and (not self.isBlackInCheckMate(currentState) and not self.isWhiteInCheckMate(currentState)):
            
            #we take the depth asigned by the method
            profunditat = depthWhite if color else depthBlack

            #recursive minamx
            valor, millors_peces_mogudes = self.recursiveminimaxGame(currentState, profunditat, color)

            #Check if there are no more moves
            if millors_peces_mogudes is None:
                print(f"No s'ha trobat cap moviment vàlid per {'Blanques' if color else 'Negres'}. (Potser Stalemate?)")
                break

            #The recursive method returna state of the movement but we need the movement itself
            moviment = self.getMovement(currentState, millors_peces_mogudes)
            
            
            if moviment[0] is None or moviment[1] is None:
                print("Error in getMovement, no move found. Aborting.")
                print("Current State:", currentState)
                print("Pieces Moved:", millors_peces_mogudes)
                break 
            
            #We separate the movement itself from the piece number id
            #If we didnt we would get an erro in movepiece, who only wnats the position
            from_pos = moviment[0][0:2]
            to_pos = moviment[1][0:2]
            self.chess.moveSim(from_pos, to_pos)
                        
            #print
            print(f"Torn de {'Blanques' if color else 'Negres'}")
            print(f"Movem {from_pos} a {to_pos} (Valor: {valor})")
            self.chess.boardSim.print_board()

            #update the state
            currentState = self.getCurrentSimState()
            color = not color

        #last print
        if self.isBlackInCheckMate(currentState):
            print("Escac i mat! Guanyen les blanques.")
        elif self.isWhiteInCheckMate(currentState):
            print("Escac i mat! Guanyen les negres.")
        else:
            print("Partida acabada (sense escac i mat).")
            if len(currentState) == 2:
                print("Nomes queden els reis")            
        return

    def recursiveAlphaBetaPoda(self, node, depth, color, alpha, beta):
        
        self.newBoardSim(node)

        #Check if kings are missing, like in minmax to prevent NoneType errors
        bkState = self.getPieceState(node, 12)
        wkState = self.getPieceState(node, 6)

        if wkState is None:
            return -float('inf') if color else float('inf'), node
            
        if bkState is None:
            return float('inf') if color else -float('inf'), node
        
        #Base case: depth limit reached
        if depth == 0:
            return self.heuristica(node, color), node

        #Check for checkmate
        if (color and self.isWhiteInCheckMate(node)) or (not color and self.isBlackInCheckMate(node)):
            if color:
                return -float('inf'), node
            else:
                return float('inf'), node 

        #Generate next states
        next_states = []
        if color:
            current_player_state = self.getWhiteState(node)
            rival_state = self.getBlackState(node)
            next_states = self.getListNextStatesW(current_player_state)
        else:
            current_player_state = self.getBlackState(node)
            rival_state = self.getWhiteState(node)
            next_states = self.getListNextStatesB(current_player_state)

        if not next_states:
            return self.heuristica(node, color), node
        
        millorValor = -float('inf')
        millorMove = None 

        for moves in next_states:
            #Check if move is legal
            is_legal, next_node = self.is_legal_transition(current_player_state, rival_state, moves, color)

            if not is_legal:
                continue 

            self.newBoardSim(next_node)

            #Recursive call 
            valor_oponent, _ = self.recursiveAlphaBetaPoda(next_node, depth - 1, not color, -beta, -alpha)
            valor_actual = -valor_oponent
            
            if valor_actual > millorValor:
                millorValor = valor_actual
                millorMove = self.copyState(next_node)

            #Update alpha to get the best value to maximizer
            alpha = max(alpha, valor_actual)
            
            #The moment alpha is bigger or equal than beta we can stop 
            if alpha >= beta:
                break

        self.newBoardSim(node)

        return millorValor, millorMove


    def alphaBetaPoda(self, depthWhite, depthBlack):
        
        #Alpha-Beta pruning game simulation for both players.
        
        color = True  #whites start
        currentState = self.getCurrentState()
        self.newBoardSim(currentState)

        #Main game loop
        while (not self.is_Draw(currentState)) and (not self.isBlackInCheckMate(currentState) and not self.isWhiteInCheckMate(currentState)):

            #Select the search depth based on the current player's turn.
            profunditat = depthWhite if color else depthBlack

            #Call the recursive function with Alpha-Beta pruning to get the best move.
            valor, next_state = self.recursiveAlphaBetaPoda(
                currentState, 
                profunditat, 
                color, 
                -float('inf'),  #inital value of alpha
                float('inf')    #inital value of beta
            )
            
            #if there no possible move may be drown
            if next_state is None:
                print(f"No s'ha trobat cap moviment vàlid per a {'Blanques' if color else 'Negres'}. (Possible ofegat?)")
                break
                
            #get the actual move
            moviment = self.getMovement(currentState, next_state)
            
            if moviment[0] is None or moviment[1] is None:
                print("Error a 'getMovement', no s'ha trobat el moviment. Avortant.")
                print("Estat Actual:", currentState)
                print("Estat Següent:", next_state)
                break 
            
            #get the origin and destination positions
            from_pos = moviment[0][0:2]
            to_pos = moviment[1][0:2]
            self.chess.moveSim(from_pos, to_pos, verbose=False)
                        
            #actual turn print
            print(f"Torn de {'Blanques' if color else 'Negres'}")
            print(f"Movem {from_pos} a {to_pos} (Valor: {valor})")
            self.chess.boardSim.print_board()
            
            #update the current state
            currentState = self.getCurrentSimState()
            color = not color

        #Final print
        if self.isBlackInCheckMate(currentState):
            print("Escac i mat! Guanyen les blanques.")
        elif self.isWhiteInCheckMate(currentState):
            print("Escac i mat! Guanyen les negres.")
        else:
            print("Partida acabada.")
            if self.is_Draw(currentState):
                print("La partida ha acabat en empat (taules).")
        return
        

    #global variable to store the initial depth of the expectimax call
    initialdepth=0

    def isChancenode(self, depth):
        if (depth % 2 == self.initialdepth % 2):
            return True
        
        return False

    def recursiveexpectimax(self, node, depth, color):

        self.newBoardSim(node)

        allmoves = []

        #This checks are made to prevent the NoneType error, that appears when a king is missing
        bkState = self.getPieceState(node, 12)
        wkState = self.getPieceState(node, 6)

        #The case the king was missing we cosnider the game has ended
        if wkState is None: #White king is gone.
            #If the current player is black, it has won and if it is white has lost
            return -float('inf') if color else float('inf'), node
            
        if bkState is None: #Black king is gone.
            #The same from befor but backwards
            return float('inf') if color else -float('inf'), node
        
        #If the limit of movemtns is reached stop the recursion
        if depth == 0:
            return self.heuristica(node, color), node
        
        if (color and self.isWhiteInCheckMate(node)) or (not color and self.isBlackInCheckMate(node)):
            #In case the current player is in checkmate, return the worst possible score depending on who whas playing
            #return-float('inf'), node
            if color:
                return -float('inf'), node
            else:
                return float('inf'), node 

        #clean the next states list
        next_states = []

        if color: #whites are moving
            current_player_state = self.getWhiteState(node)
            rival_state = self.getBlackState(node)
            next_states = self.getListNextStatesW(current_player_state)
        else: #blacks are moving
            current_player_state = self.getBlackState(node)
            rival_state = self.getWhiteState(node)
            next_states = self.getListNextStatesB(current_player_state)

        #In case no one can move
        if not next_states:
            return self.heuristica(node, color), node
       
        # Always looking from whites perspective
        millorValor = -float('inf')
        millorMove = None

        for moves in next_states:
            is_legal, next_node = self.is_legal_transition(current_player_state, rival_state, moves, color)

            if not is_legal:
                continue 
            #state to prevenet modifying overwriting
            self.newBoardSim(next_node)

            valor_oponent, _ = self.recursiveexpectimax(next_node, depth - 1, not color)

            valor_actual = -valor_oponent

            if self.isChancenode(depth):
                allmoves.append(valor_actual)

            if valor_actual > millorValor:
                millorValor = valor_actual
                millorMove = self.copyState(next_node)
            
        if millorMove is None and next_states:
            millorMove = next_states[0]

        #restore the state
        self.newBoardSim(node)

        if self.isChancenode(depth):
            if not allmoves:
                return self.heuristica(node, color), node
            
            #calc the value of the expected value of the rival
            expected_value = self.calculateValue(allmoves)

            #chance nodesodenst hace to choose a move, but I will return the best one   
            return expected_value, millorMove
        else:
            #return the best move for the maximizing player
            if millorMove is None and next_states:
                millorMove = next_states[0]
            return millorValor, millorMove  

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
    ##########################
    #Mixed game for question 3
    ##########################
    #blacks use alpha-beta and whites use minimax
    def alfa_black_minmax_white(self, depthWhite, depthBlack):
        #Mixed game simulation: White uses Minimax, Black uses Alpha-Beta Pruning.
        color = True  #True = White (Expectimax), False = Black (Alpha-Beta)
        currentState = self.getCurrentState()
        self.newBoardSim(currentState)
        
        while (not self.is_Draw(currentState)) and (not self.isBlackInCheckMate(currentState) and not self.isWhiteInCheckMate(currentState)):
            #We are gonna change of algorithm everytime the turn alterantes, so the white = color true
            #we take the depth asigned by the method
            profunditat = depthWhite if color else depthBlack

            #Vefore defininf valor and millors_peces_mogudes, we initialize them in none
            valor = None
            millors_peces_mogudes = None

            #after that depending on the turn we call one or another algorithm
            #recursive minamx
            if color:
                # White's Turn: Expectimax
                self.initialdepth = profunditat # Set initial depth for isChancenode logic
                valor, millors_peces_mogudes = self.recursiveminimaxGame(currentState, profunditat, color)
            else:
                # Black's Turn: Alpha-Beta Pruning
                valor, millors_peces_mogudes = self.recursiveAlphaBetaPoda(currentState, profunditat, color, -float('inf'), float('inf'))
            
            if millors_peces_mogudes is None:
                print(f"No s'ha trobat cap moviment vàlid per {'Blanques' if color else 'Negres'}. (Potser Stalemate?)")
                break
            #The recursive method returna state of the movement but we need the movement itself
            moviment = self.getMovement(currentState, millors_peces_mogudes)
            
            if moviment[0] is None or moviment[1] is None:
                print("Error in getMovement, no move found. Aborting.")
                print("Maybe is draw?")
                print("Current State:", currentState)
                print("Pieces Moved:", millors_peces_mogudes)
                break 
            
            #We separate the movement itself from the piece number id
            from_pos = moviment[0][0:2]
            to_pos = moviment[1][0:2]
            self.chess.moveSim(from_pos, to_pos)
                        
            #print
            print(f"Torn de {'Blanques' if color else 'Negres'}")
            print(f"Movem {from_pos} a {to_pos} (Valor: {valor})")
            self.chess.boardSim.print_board()

            #update the state
            currentState = self.getCurrentSimState()
            color = not color

        #last print
        if self.isBlackInCheckMate(currentState):
            print("Escac i mat! Guanyen les blanques.")
        elif self.isWhiteInCheckMate(currentState):
            print("Escac i mat! Guanyen les negres.")
        else:
            print("Partida acabada (sense escac i mat).")
            if len(currentState) == 2:
                print("Nomes queden els reis")            
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



    def translate_action(self, action, currentState):
        """
        Calcula el nou estat basat en l'acció donada.
        Actions: 0: Up, 1: Down, 2: Right, 3: Left
        State: (row, column, piece)
        """
        row, col, piece = currentState

        # Coordenades de matriu: (0,0) és Top-Left
        if action == 0:   # Up
            row -= 1
        elif action == 1: # Down
            row += 1
        elif action == 2: # Right
            col += 1
        elif action == 3: # Left
            col -= 1
        
        # Retornem la tupla actualitzada
        return (row, col, piece)




    def aiGame(self):        
        currentState = self.getCurrentState()    
        color = True
        self.newBoardSim(currentState)

        #main game loop
        while (not self.is_Draw(currentState)) and (not self.isBlackInCheckMate(currentState) and not self.isWhiteInCheckMate(currentState)):


            if color:
                #white move minimax
                env.reset_environment()
                self.ai.reduce_exploration_rate_by_decrease_rate()
                state = self.env.get_state()
                done = False

        while not done:
            # Fem acció
            action = self.ai.think(state)

            # Movem a env i chess
            next_state = self.translate_action(action, currentState)

            #CHESS
            moviment = self.getMovement(self.getCurrentSimState())
            if moviment[0] is None or moviment[1] is None:
                print("Error a 'getMovement', no s'ha trobat el moviment. Avortant.")
                print("Estat actual:", currentState)
                break 
            
            #execute the movement
            from_pos = moviment[0][0:2]
            to_pos = moviment[1][0:2]
            self.chess.moveSim(from_pos, to_pos)      

            #env
            next_state, reward, done = env.move_piece(action)
            self.ai.learn(state, action, reward, next_state, done)
            state = next_state

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




if __name__ == "__main__":
    # Initialize an empty 8x8 chess board
    TA = np.zeros((8, 8))


    # Load initial positions of the pieces
    TA = np.zeros((8, 8))
    TA[7][3] = 6     
    TA[0][4] = 12  
    aichess = Aichess(TA, True)
    print("printing board")
    aichess.chess.boardSim.print_board()

    for i in range(1000):
        aichess.aiGame()
    
    
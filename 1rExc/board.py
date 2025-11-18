import piece  # Asumeixo que tens aquest mòdul del teu codi original
import numpy as np

class Board:
    """
    Representació d'un tauler d'escacs reduït (3x4) per a entrenament RL.
    """
    initState
    
    rows
    cols
        
    board
    currentStateW
    currentStateB
    listSuccessorStates

    def __init__(self, initState=None):
        self.rows = 3
        self.cols = 4
        
        self.board = []
        self.currentStateW = []
        self.currentStateB = []
        self.listSuccessorStates = []

        # Inicialització del tauler buit (Matriu 3x4)
        for i in range(self.rows):
            self.board.append([None] * self.cols)

        if initState is None:
            # Configuració per defecte si no es passa estat inicial
            # Exemple: Rei blanc a (0,0)
            self.board[0][0] = piece.King(True) 
            self.currentStateW.append([0, 0, 6]) # 6 = ID del Rei
        else:
            self.parse_init_state(initState)

    def parse_init_state(self, initState):
        """Carrega l'estat des d'una matriu numèrica."""
        for i in range(self.rows):
            for j in range(self.cols):
                val = initState[i][j]
                if val == 0:
                    continue
                
                # Exemple simplificat d'assignació (Blanc)
                if val == 6:
                    self.board[i][j] = piece.King(True)
                    self.currentStateW.append([i, j, val])
                elif val == 1:
                    self.board[i][j] = piece.Pawn(True)
                    self.currentStateW.append([i, j, val])
                # ... (Afegeix aquí la resta de peces si cal) ...
                
                # Exemple (Negre/Objectiu)
                elif val == 12:
                    self.board[i][j] = piece.King(False)
                    self.currentStateB.append([i, j, val])

    def print_board(self):
        """
        Dibuixa el tauler 3x4 per consola.
        """
        print("   " + " ".join([f"{j}" for j in range(self.cols)]))
        print("  " + "-" * (self.cols * 2 + 1))
        
        for i in range(self.rows):
            row_str = f"{i}| "
            for j in range(self.cols):
                p = self.board[i][j]
                if p is None:
                    row_str += ". "
                else:
                    # Assumeixo que la classe Piece té un mètode o atribut per al símbol
                    # Si no, faig servir la inicial del nom de la classe
                    symbol = p.__class__.__name__[0] 
                    if not p.color: symbol = symbol.lower() # Minúscula per negres
                    row_str += f"{symbol} "
            print(row_str)
        print("\n")

    def isSameState(self, a, b):
        """Compara si dos estats (llistes de posicions) són idèntics."""
        if len(a) != len(b): return False
        for item in a:
            if item not in b: return False
        return True

    def getListNextStatesW(self, mypieces):
        """
        Genera els estats successors vàlids en un tauler 3x4.
        """
        self.listSuccessorStates = []

        for mypiece in mypieces:
            r, c, type_id = mypiece
            
            listPotentialMoves = []

            # --- Lògica per al REI (King) ---
            if type_id == 6: # ID del Rei
                offsets = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)
                ]
                for dr, dc in offsets:
                    listPotentialMoves.append([r + dr, c + dc, 6])

            # --- Lògica per al PEÓ (Pawn) - simplificat ---
            elif type_id == 1: # ID del Peó
                # Només mou endavant
                listPotentialMoves.append([r + 1, c, 1])

            # --- Validació de límits (Boundary Check) ---
            for move in listPotentialMoves:
                nr, nc, ntype = move
                
                # Comprovació CRÍTICA: Estem dins el tauler 3x4?
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    
                    # Comprovem si la casella està buida o ocupada per enemic
                    # (Aquí pots afegir la lògica de captura si cal)
                    if self.board[nr][nc] is None:
                        self.listSuccessorStates.append(move)
                    elif not self.board[nr][nc].color: # Si és peça negra
                        self.listSuccessorStates.append(move)

        return self.listSuccessorStates
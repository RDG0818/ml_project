import pyspiel
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state                  # Current game state at this node.
        self.parent = parent                # Parent node (None for root).
        self.move = move                    # Move that led to this node.
        self.children = {}                  # Dictionary: move -> child node.
        self.visits = 0                     # Number of times this node was visited.
        self.value = 0.0                    # Accumulated value (from simulations).
        # Store the player who is to move at this state.
        self.player = state.current_player()

    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        if self.is_terminal():
            return True
        return len(self.children) == len(self.state.legal_actions())

    def best_child(self, c_param=1.4):
        best_value = -float('inf')
        best_nodes = []
        for move, child in self.children.items():
            uct_value = (child.value / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_nodes = [child]
            elif uct_value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes) if best_nodes else None

def tree_policy(node):
    """Navigate the tree until reaching a node that is not fully expanded or a terminal state."""
    while not node.state.is_terminal():
        if not node.is_fully_expanded():
            return expand(node)
        else:
            node = node.best_child()
    return node

def expand(node):
    """Expand one of the untried moves and return the new child node."""
    legal_moves = node.state.legal_actions()
    untried_moves = [move for move in legal_moves if move not in node.children]
    move = random.choice(untried_moves)
    next_state = node.state.child(move)
    child_node = MCTSNode(next_state, parent=node, move=move)
    node.children[move] = child_node
    return child_node

def default_policy(state, root_player):
    """
    Simulate a playout until a terminal state is reached.
    This version first checks for an immediate win for the current player.
    If no immediate win is found, it then selects only among moves
    that do not immediately allow an opponent win.
    Returns the outcome from the perspective of root_player.
    """
    rollout_state = state.clone()
    while not rollout_state.is_terminal():
        legal_moves = rollout_state.legal_actions()
        move_to_play = None
        current = rollout_state.current_player()
        opponent = 1 - current

        # 1. Check for an immediate win for the current player.
        for move in legal_moves:
            test_state = rollout_state.child(move)
            if test_state.is_terminal():
                returns = test_state.returns()
                if returns[current] > 0:
                    move_to_play = move
                    break

        # 2. If no immediate win, choose a move that does not allow an immediate opponent win.
        if move_to_play is None:
            safe_moves = []
            for move in legal_moves:
                test_state = rollout_state.child(move)
                # First, if this move itself is terminal, check if it results in a loss.
                if test_state.is_terminal():
                    returns = test_state.returns()
                    if returns[opponent] > 0:
                        continue  # This move immediately loses; not safe.
                else:
                    # Otherwise, simulate each opponent move.
                    opp_immediate_win = False
                    for opp_move in test_state.legal_actions():
                        opp_test_state = test_state.child(opp_move)
                        if opp_test_state.is_terminal():
                            returns = opp_test_state.returns()
                            if returns[opponent] > 0:
                                opp_immediate_win = True
                                break
                    if opp_immediate_win:
                        continue  # This move allows the opponent to win immediately.
                safe_moves.append(move)
            if safe_moves:
                move_to_play = random.choice(safe_moves)
            else:
                move_to_play = random.choice(legal_moves)
        rollout_state.apply_action(move_to_play)
    returns = rollout_state.returns()
    return returns[root_player]

def backup(node, reward):
    """
    Backpropagate the simulation result up the tree.
    The reward is negated at each step to account for the adversarial nature of the game.
    """
    while node is not None:
        node.visits += 1
        node.value += reward
        reward = -reward  # Switch perspective for the opponent.
        node = node.parent

def mcts(root_state, iterations):
    """
    Run MCTS starting from root_state for a given number of iterations.
    Returns the move that leads to the child with the highest visit count.
    """
    root = MCTSNode(root_state)
    root_player = root.state.current_player()  # Save the root player's id.
    for _ in range(iterations):
        leaf = tree_policy(root)
        reward = default_policy(leaf.state, root_player)
        backup(leaf, reward)
    best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_move

def play_game(iterations_per_move=50000):
    """Play a complete game of tic tac toe with MCTS playing for both players."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    
    print("Starting Tic Tac Toe (MCTS vs. MCTS):")
    while not state.is_terminal():
        print("\nCurrent state:")
        print(state)  # Prints a human-readable board.
        move = mcts(state, iterations_per_move)
        print("MCTS selected move:", move)
        state.apply_action(move)
    
    print("\nFinal state:")
    print(state)
    outcome = state.returns()
    print("Game over. Outcome (player 0, player 1):", outcome)

if __name__ == "__main__":
    play_game()

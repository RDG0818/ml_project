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
        # Store the player who is to move at this state. This is used to evaluate rewards.
        self.player = state.current_player()

    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        if self.is_terminal():
            return True
        # Compare the moves we’ve expanded with all legal moves.
        return len(self.children) == len(self.state.legal_actions())

    def best_child(self, c_param=1.4):
        """Select the child with the highest UCT value."""
        best_value = -float('inf')
        best_nodes = []
        for move, child in self.children.items():
            # UCT (Upper Confidence Bound applied to Trees) formula.
            uct_value = (child.value / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_nodes = [child]
            elif uct_value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes) if best_nodes else None

def tree_policy(node):
    """Selection and expansion: navigate the tree until we reach a non-fully expanded node or terminal state."""
    while not node.state.is_terminal():
        if not node.is_fully_expanded():
            return expand(node)
        else:
            node = node.best_child()
    return node

def expand(node):
    """Expand one of the untried moves and return the new child node."""
    legal_moves = node.state.legal_actions()
    # Determine which moves haven't been expanded yet.
    untried_moves = [move for move in legal_moves if move not in node.children]
    move = random.choice(untried_moves)
    next_state = node.state.child(move)
    child_node = MCTSNode(next_state, parent=node, move=move)
    node.children[move] = child_node
    return child_node

def default_policy(state, player):
    """
    Simulate a random playout (rollout) until a terminal state is reached.
    The return value is the payoff from the perspective of 'player'.
    """
    # Copy the state so as not to affect the tree.
    rollout_state = state.clone()
    while not rollout_state.is_terminal():
        legal_moves = rollout_state.legal_actions()
        move = random.choice(legal_moves)
        rollout_state.apply_action(move)
    # In zero-sum games like tic tac toe, the returns are from the perspective of both players.
    # We extract the reward for the player who started the simulation.
    returns = rollout_state.returns()
    return returns[player]

def backup(node, reward):
    """
    Backpropagate the simulation result up the tree.
    At each step the reward is negated, reflecting the adversarial nature of the game.
    """
    while node is not None:
        node.visits += 1
        node.value += reward
        reward = -reward  # Negate the reward as we move to the opponent's perspective.
        node = node.parent

def mcts(root_state, iterations):
    """
    Run MCTS starting from root_state for a given number of iterations.
    Returns the move (action) that leads to the child with the highest visit count.
    """
    root = MCTSNode(root_state)
    for _ in range(iterations):
        # Selection & Expansion.
        leaf = tree_policy(root)
        # The simulation is run from the leaf node's state,
        # and we use the player stored in the leaf node for reward evaluation.
        reward = default_policy(leaf.state, leaf.player)
        # Backpropagation with reward negation.
        backup(leaf, reward)
    # Choose the move that was most visited (i.e. the most promising).
    best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_move

def play_game(iterations_per_move=1000):
    """Play a complete game of tic tac toe with MCTS on both sides."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    
    print("Starting Tic Tac Toe (MCTS vs. MCTS):")
    while not state.is_terminal():
        print("\nCurrent state:")
        print(state)  # This prints a human-readable board.
        move = mcts(state, iterations_per_move)
        print("MCTS selected move:", move)
        state.apply_action(move)
    
    print("\nFinal state:")
    print(state)
    outcome = state.returns()
    print("Game over. Outcome (player 0, player 1):", outcome)

if __name__ == "__main__":
    play_game()

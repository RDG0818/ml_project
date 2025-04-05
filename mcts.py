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
        self.value = 0.0                    # Accumulated value (negated reward propagated up)
        self.player = state.current_player() # Player to move *at this node*

    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        if self.is_terminal():
            return True
        # Ensure state is not None and has legal_actions method
        if self.state is None or not hasattr(self.state, 'legal_actions'):
             # Handle cases like node representing a terminal state maybe?
             # Or if state creation failed. For robustness:
             return True # Treat as fully expanded if state invalid
        try:
             legal_moves = self.state.legal_actions()
             return len(self.children) == len(legal_moves)
        except Exception as e:
             print(f"Error accessing legal_actions for state: {self.state}. Error: {e}")
             # Depending on the game/error, might need specific handling.
             # For now, treat as expanded to avoid crashing search.
             return True


    def best_child(self, c_param=1.4):
        best_value = -float('inf')
        best_nodes = []

        # Ensure parent visits > 0 before taking log. This should generally be true
        # if best_child is called from tree_policy after the node has been visited.
        if self.visits == 0:
            # This case should ideally not happen if called correctly, but handle defensively.
            # Perhaps return a random child or None?
            # Or if children exist, maybe pick one randomly? For now:
            return random.choice(list(self.children.values())) if self.children else None

        log_parent_visits = math.log(self.visits)

        for child in self.children.values(): # Iterate directly over child nodes
            if child.visits == 0:
                # Prioritize exploring unvisited children - assign effectively infinite value
                uct_value = float('inf')
            else:
                # ****** THE KEY CHANGE ******
                # child.value / child.visits = avg score for the player who MOVED TO child (opponent)
                # -child.value / child.visits = avg score for the player AT THIS NODE (self)
                exploitation_term = (-child.value / child.visits)
                exploration_term = c_param * math.sqrt(log_parent_visits / child.visits)
                uct_value = exploitation_term + exploration_term

            if uct_value > best_value:
                best_value = uct_value
                best_nodes = [child]
            elif uct_value == best_value:
                # Handle ties by collecting all tied best nodes
                best_nodes.append(child)

        # Choose randomly among the best nodes found
        return random.choice(best_nodes) if best_nodes else None


def tree_policy(node):
    """Navigate the tree using UCT until reaching a node that is not fully expanded or a terminal state."""
    current_node = node
    while not current_node.state.is_terminal():
        if not current_node.is_fully_expanded():
            return expand(current_node)
        else:
            # Ensure best_child doesn't return None if node is fully expanded but has children
            next_node = current_node.best_child()
            if next_node is None:
                 # This might happen if all children have visits=0 and best_child doesn't handle it perfectly,
                 # or some other unexpected state. Fallback or raise error.
                 # If it's fully expanded, it should have children that have been visited or explored.
                 # Let's pick a random one if selection fails unexpectedly.
                 print(f"Warning: best_child returned None for fully expanded node. State:\n{current_node.state}")
                 if current_node.children:
                      next_node = random.choice(list(current_node.children.values()))
                 else:
                      # Node is fully expanded but has no children? This implies no legal moves?
                      # Should have been caught by is_terminal earlier. Return current node.
                      return current_node # It's effectively terminal for the search path
            current_node = next_node
            if current_node is None: # Defensive check after assignment
                 # This indicates a deeper issue if it occurs.
                 raise Exception("MCTS Error: Failed to select or expand node in tree_policy.")
    return current_node # Return the terminal node

def expand(node):
    """Expand one of the untried moves and return the new child node."""
    legal_moves = node.state.legal_actions()
    untried_moves = [move for move in legal_moves if move not in node.children]

    # Ensure there are untried moves before choosing
    if not untried_moves:
        # This shouldn't happen if called after checking !is_fully_expanded()
        # but handle defensively. Could indicate a state inconsistency.
        print(f"Warning: expand called on fully expanded node? State:\n{node.state}")
        # Perhaps return the node itself or raise an error?
        return node # Cannot expand further

    move = random.choice(untried_moves)
    try:
        next_state = node.state.child(move)
        child_node = MCTSNode(next_state, parent=node, move=move)
        node.children[move] = child_node
        return child_node
    except Exception as e:
        print(f"Error creating child state for move {move}. State:\n{node.state}\nError: {e}")
        # Cannot expand this path, maybe return the original node?
        return node


# Use the simple random default policy
def default_policy_random(state, root_player):
    """
    Simulate a purely random playout until a terminal state is reached.
    Returns the outcome (+1 Win, 0 Draw, -1 Loss) from the perspective of root_player.
    """
    rollout_state = state.clone()
    while not rollout_state.is_terminal():
        action = random.choice(rollout_state.legal_actions())
        rollout_state.apply_action(action)

    returns = rollout_state.returns()
    # Ensure returns are normalized to +1, 0, -1 for consistency
    score = returns[root_player]
    if score > 0:
        return 1.0
    elif score < 0:
        return -1.0
    else:
        return 0.0

def backup(node, reward):
    """
    Backpropagate the simulation result (relative to root_player) up the tree.
    The reward is negated at each step. This means node.value accumulates
    value relative to the player who MOVED TO this node state.
    """
    current_node = node
    current_reward = reward
    while current_node is not None:
        current_node.visits += 1
        # Add the reward perspective from the level below
        current_node.value += current_reward
        # Negate reward for the parent (switching player perspective)
        current_reward = -current_reward
        current_node = current_node.parent


def mcts(root_state, iterations):
    """
    Run MCTS starting from root_state for a given number of iterations.
    Returns the move that leads to the child with the highest visit count.
    """
    if root_state.is_terminal():
        print("Warning: MCTS called on terminal state.")
        return None # No move possible

    root = MCTSNode(root_state.clone()) # Clone to avoid modifying the original state
    root_player = root.state.current_player()

    if iterations == 0:
         # Handle 0 iterations: return random legal move
         legal_moves = root.state.legal_actions()
         return random.choice(legal_moves) if legal_moves else None

    for i in range(iterations):
        # Add check for root terminal state potentially reached during search
        # (though unlikely with standard MCTS structure)
        if root.state.is_terminal():
             break
        leaf = tree_policy(root)
        # Ensure leaf is not None (can happen if tree_policy has issues)
        if leaf is None:
             print(f"Warning: tree_policy returned None at iteration {i}. Root state:\n{root.state}")
             continue # Skip this iteration or handle error

        # Perform rollout only if the leaf is not already terminal
        if not leaf.state.is_terminal():
             reward = default_policy_random(leaf.state, root_player)
        else:
             # If leaf is terminal, get reward directly
             returns = leaf.state.returns()
             score = returns[root_player]
             if score > 0: reward = 1.0
             elif score < 0: reward = -1.0
             else: reward = 0.0

        backup(leaf, reward)

    # Ensure root has children before selecting best move
    if not root.children:
        # If no children after iterations, it means either:
        # 1. No legal moves from root (handled at start)
        # 2. All simulations failed to expand? (Error)
        # 3. Iterations was 0 (handled)
        # 4. tree_policy/expand issues
        print(f"Warning: Root node has no children after {iterations} iterations. State:\n{root.state}")
        # Fallback to random move if possible
        legal_moves = root.state.legal_actions()
        return random.choice(legal_moves) if legal_moves else None

    # Choose the move corresponding to the most visited child node
    # This is generally more robust than choosing based on value
    best_move_node = max(root.children.values(), key=lambda node: node.visits)
    return best_move_node.move


def play_game(iterations_per_move=50000):
    """Play a complete game of tic tac toe with MCTS playing for both players."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()

    print("Starting Tic Tac Toe (MCTS vs. MCTS):")
    turn = 0
    while not state.is_terminal():
        print(f"\nTurn {turn+1}, Player {state.current_player()}")
        print("Current state:")
        print(state)
        move = mcts(state, iterations_per_move)

        if move is None:
             print("MCTS returned no move. Game ending.")
             break

        print(f"MCTS selected move: {move} (Action ID)")

        # Optional: Map action ID to a more readable format if desired
        # print(f"MCTS selected move: {state.action_to_string(state.current_player(), move)}")

        state.apply_action(move)
        turn += 1

    print("\nFinal state:")
    print(state)
    outcome = state.returns()
    print(f"Game over. Outcome (player 0, player 1): {outcome}")
    if outcome[0] == 1: print("Player 0 wins!")
    elif outcome[1] == 1: print("Player 1 wins!")
    else: print("It's a draw!")

if __name__ == "__main__":
    # Reduce iterations for faster testing, increase again for stronger play
    play_game(iterations_per_move=100000) # Start with fewer, e.g., 10k
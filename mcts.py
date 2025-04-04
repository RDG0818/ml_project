import math
import random
import pyspiel
import sys # For float('inf') if needed, though math.inf is preferred

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state         # an OpenSpiel state object
        self.parent = parent
        self.move = move           # The move that led to this state
        self.player_at_node = state.current_player() if not state.is_terminal() else (1 - parent.player_at_node) # Player whose turn it IS at this node. Handle terminal case.
        self.children = {}         # map move to Node
        self.visit_count = 0
        self.total_value = 0.0     # Accumulated value *from the perspective of player_at_node*

    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        if self.is_terminal():
            return True
        legal_moves = self.state.legal_actions()
        return len(self.children) == len(legal_moves)

    def value(self):
        """ Returns the average value of this node from the perspective
            of the player whose turn it is at this node. """
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

class MCTS:
    # Use exploration_weight=sqrt(2) as it's common, and gamma=1.0 for TicTacToe
    def __init__(self, exploration_weight=math.sqrt(2), gamma=1.0):
        """
        exploration_weight (C_p): controls the trade-off between exploration and exploitation.
        gamma: discount factor (set to 1.0 for undiscounted terminal rewards).
        """
        self.exploration_weight = exploration_weight
        self.gamma = gamma # Should be 1.0 for TicTacToe

    def search(self, initial_state, num_simulations):
        root = Node(initial_state)
        # Keep track of the root player just for the rollout perspective, although
        # the negamax approach internalizes player perspectives.
        root_player_for_rollout = initial_state.current_player()

        if root.is_terminal():
             raise ValueError("Cannot search from a terminal state.")

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # 1. Selection: Traverse the tree using UCB1 variant for alternating players.
            while node.is_fully_expanded() and not node.is_terminal():
                # Selection should only happen if fully expanded and not terminal
                move, node = self._select_child(node)
                search_path.append(node)

            # 2. Expansion: If the node is non-terminal and not fully expanded, expand one child.
            # This check implicitly handles the case where Selection stops at a non-fully-expanded node.
            if not node.is_fully_expanded() and not node.is_terminal():
                node = self._expand(node)
                search_path.append(node) # Add the newly expanded node to the path

            # 3. Simulation (Rollout): Run a random simulation from the new node (or terminal node).
            #    The reward is calculated from the perspective of the root player of the *entire search*.
            reward = self._rollout(node.state, root_player_for_rollout)

            # 4. Backpropagation: Update nodes along the path, alternating reward sign.
            self._backpropagate(search_path, reward)

        # Choose the best move from the root based on negamax principle
        # The root wants to maximize its own value. Its children's values are from
        # the opponent's perspective. So root maximizes -child.value().
        # This is equivalent to choosing the child with the minimum value.
        # As an alternative robustness measure, often the most visited child is chosen.
        # Let's choose based on value first:
        best_move = None
        best_value = -float('inf') # Root wants to maximize its value (-child_value)

        if not root.children:
             print("Warning: Root has no children after search. Choosing random move.")
             # This might happen if num_simulations is very small or root is terminal (already checked)
             return random.choice(initial_state.legal_actions()), 0.0

        for move, child in root.children.items():
             # Value from the root's perspective is -child.value()
             # because child.value() is from the opponent's perspective.
             current_value = -child.value()
             # print(f"Move {move}: ChildValue={child.value():.3f}, Visits={child.visit_count}, RootPerspectiveValue={current_value:.3f}") # Debug print
             if current_value > best_value:
                 best_value = current_value
                 best_move = move

        # Alternative: Choose most visited child (often more stable)
        # best_move = max(root.children, key=lambda m: root.children[m].visit_count)
        # best_value = -root.children[best_move].value() # Value from root's perspective

        if best_move is None:
            print("Error: Could not determine best move. Choosing random.")
            # Fallback if all children have same value (e.g., 0) and first check fails.
            best_move = random.choice(list(root.children.keys()))
            best_value = -root.children[best_move].value()


        return best_move, best_value

    def _select_child(self, node):
        """ Selects a child node using UCB1 formula adapted for negamax. """
        # Parent visit count MUST be positive if we are selecting a child from it
        # after it has been visited at least once (which led to its expansion).
        log_parent_visits = math.log(node.visit_count)

        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in node.children.items():
            if child.visit_count == 0:
                # Prioritize unvisited children (infinite score) - should ideally be handled by expansion logic
                # If we reach here for a node considered 'fully expanded', it's slightly unusual but handle it.
                score = float('inf')
            else:
                # child.value() is from the child's perspective (opponent)
                # Parent wants to maximize: -child.value() + exploration
                exploitation = -child.value()
                exploration = self.exploration_weight * math.sqrt(log_parent_visits / child.visit_count)
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        if best_child is None:
             # This should not happen in a node that is fully_expanded
             print(f"Warning: _select_child failed to find best child for node with {node.visit_count} visits. State:\n{node.state}")
             # Fallback to random choice among existing children
             if node.children:
                 best_move = random.choice(list(node.children.keys()))
                 best_child = node.children[best_move]
             else:
                 # This indicates a deeper issue - selecting from a node with no children that thinks it's expanded
                 raise RuntimeError("Select child called on fully expanded node with no children.")


        return best_move, best_child

    def _expand(self, node):
        """ Expands one untried child node. """
        legal_moves = node.state.legal_actions()
        untried_moves = [m for m in legal_moves if m not in node.children]

        if not untried_moves:
            raise RuntimeError("Expand called on fully expanded node")

        move = random.choice(untried_moves)
        next_state = node.state.clone()
        next_state.apply_action(move)
        child_node = Node(next_state, parent=node, move=move)
        node.children[move] = child_node
        return child_node


    def _rollout(self, state, root_player):
        """ Simulates a random playout from the state.
            Returns the terminal reward from the perspective of root_player.
        """
        current_state = state.clone()
        while not current_state.is_terminal():
            legal_moves = current_state.legal_actions()
            # Handle cases like breakthrough where no moves might be possible (though not TicTacToe)
            if not legal_moves:
                return 0 # Or handle based on game rules (e.g., loss for current player)
            move = random.choice(legal_moves)
            current_state.apply_action(move)

        # Get the terminal reward from the perspective of the original root player.
        terminal_reward = current_state.returns()[root_player]
        return terminal_reward


    def _backpropagate(self, search_path, reward):
        """ Backpropagates the reward up the path, negating for alternating players. """
        # The reward from rollout is from root_player's perspective.
        # We need to adjust it based on the player at each node.
        # Let current_reward be the value from the perspective of the *child* node's player.

        current_reward = reward # Start with reward from root_player perspective

        for node in reversed(search_path):
            # The value stored in the node should be from the perspective of node.player_at_node.
            # The `current_reward` coming up is from the perspective of the player
            # at the *child* of this node (or from root_player for the leaf).
            # Check if the node's player matches the perspective of the reward.
            # If node.player_at_node == root_player, reward matches perspective.
            # If node.player_at_node != root_player, reward is opponent's perspective.
            # A simpler way for zero-sum games: negate at each step.
            # The `reward` passed *into* the update for `node` should represent the outcome
            # achieved by the player who moved *to* `node`.

            node.visit_count += 1
            # Add the value from the perspective of the player AT THIS NODE.
            # Since the player alternates, the reward seen by this node is the
            # negative of the reward seen by the child.
            # Let's refine: `reward` is passed up. It represents the value from the perspective
            # of the player who MOVED TO THE CHILD. This player is node.player_at_node.
            # So, directly add the incoming reward? No, that's not quite right for negamax backprop.

            # Correct Negamax Backprop: The value stored should be from the perspective of
            # node.player_at_node. The reward from the simulation needs to be potentially flipped
            # based on who node.player_at_node is relative to root_player.
            # Then, as it propagates, it flips sign each time.

            # Let's try the simple negation approach first:
            node.total_value += reward # Add the reward seen from the *child's* move perspective
            reward *= -1 # Negate reward for the parent (alternating player)
            reward *= self.gamma # Apply discount factor AFTER negation

def play_full_game():
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    # Use gamma=1.0 for Tic Tac Toe, sqrt(2) is a common exploration constant
    mcts = MCTS(exploration_weight=math.sqrt(2), gamma=1.0)

    print("Starting Tic Tac Toe Game (Negamax MCTS)")
    turn = 0
    while not state.is_terminal():
        print("-" * 20)
        print(f"Turn {turn}, Player {state.current_player()} to move:")
        print(state)

        simulations = 10000 if turn < 6 else 20000 # Maybe fewer sims early game? Adjust as needed.
        print(f"Running {simulations} MCTS simulations...")
        best_move, best_value = mcts.search(state, num_simulations=simulations)

        # The returned best_value is from the current player's perspective
        print(f"Player {state.current_player()} selects move: {best_move} (predicted value: {best_value:.4f})\n")

        # Check if move is legal before applying (debugging sanity check)
        if best_move not in state.legal_actions():
            print(f"ERROR: MCTS chose illegal move {best_move}!")
            print(f"Legal moves were: {state.legal_actions()}")
            break # Stop the game

        state.apply_action(best_move)
        turn += 1

    print("=" * 20)
    print("Game Over!")
    print("Final state:")
    print(state)
    returns = state.returns()
    print("Returns:", returns)
    if returns[0] == 1:
        print("Player 0 Wins!")
    elif returns[1] == 1:
        print("Player 1 Wins!")
    else:
        print("It's a Draw!")

if __name__ == "__main__":
    play_full_game()
import math
import random
import pyspiel
import time

class MCTSNode:
    """
    Represents a node in the Monte Carlo Search Tree.
    Stores statistics for the associated game state.
    """
    def __init__(self, state, parent=None, move=None):
        self.state = state  # The PySpiel state object corresponding to this node
        self.parent = parent
        self.move = move    # The move that led from the parent to this node

        # Player whose turn it is AT THIS NODE's state.
        # Need to handle terminal states where current_player might be invalid.
        self.player_to_move = -1 # Placeholder
        if not self.state.is_terminal():
            self.player_to_move = self.state.current_player()
        elif parent:
            # If terminal, the "player to move" concept is less meaningful,
            # but for value assignment, consider it the opponent of the player who just moved.
             self.player_to_move = 1 - parent.player_to_move
        # else: root is terminal - should be handled before search starts.

        self.children = {}  # Maps action -> MCTSNode
        self.visit_count = 0
        # Q-value: Sum of outcomes FOR THE PLAYER TO MOVE AT THIS NODE
        # from simulations passing through this node.
        self.total_value = 0.0

        # Store legal actions and track untried ones for expansion
        self.legal_actions = self.state.legal_actions() if not self.state.is_terminal() else []
        self.untried_actions = list(self.legal_actions) # Keep a mutable list

    def is_fully_expanded(self):
        """Does this node have children for all its legal actions?"""
        return not self.untried_actions and self.legal_actions # True if no untried actions and there were legal actions

    def is_terminal(self):
        """Is the state at this node terminal?"""
        return self.state.is_terminal()

    def get_value_estimate(self):
        """Returns the average outcome for the player_to_move at this node."""
        if self.visit_count == 0:
            return 0.0  # Default value for unvisited nodes
        return self.total_value / self.visit_count

class MCTS:
    """
    Monte Carlo Tree Search algorithm implementation for two-player zero-sum games.
    Uses the UCB1 formula (adapted for negamax) for selection.
    """
    def __init__(self, exploration_constant=math.sqrt(2)):
        self.C = exploration_constant  # UCB exploration constant

    def search(self, initial_state, num_simulations):
        """
        Performs MCTS search from the initial_state for a given number of simulations.
        Returns the best move found.
        """
        if initial_state.is_terminal():
            raise ValueError("Cannot run search on a terminal state.")

        root = MCTSNode(initial_state)

        if not root.legal_actions:
             raise ValueError("Cannot run search on a state with no legal actions (should be terminal).")

        start_time = time.time()
        for i in range(num_simulations):
            # print(f"Sim {i+1}/{num_simulations}") # Debug: Track simulation progress
            node = root
            path = [node] # Path for backpropagation

            # 1. Selection Phase: Traverse down the tree using UCB1 until an expandable or terminal node is reached.
            while node.is_fully_expanded() and not node.is_terminal():
                node = self._select_best_uct_child(node)
                path.append(node)
            # print(f"  Selection ended at node for player {node.player_to_move}, terminal={node.is_terminal()}, fully_expanded={node.is_fully_expanded()}")

            # 2. Expansion Phase: If the node is not terminal, expand one untried action.
            if not node.is_terminal(): # Implicitly, if it's not fully expanded
                node = self._expand(node)
                path.append(node)
            # print(f"  Expansion resulted in node for player {node.player_to_move}, move={node.move}")

            # 3. Simulation Phase (Rollout): Simulate a random playout from the new node (or selected terminal node).
            reward = self._rollout(node.state)
            # print(f"  Rollout reward: {reward}") # reward is array [P0_outcome, P1_outcome]

            # 4. Backpropagation Phase: Update visit counts and values along the path from the leaf to the root.
            self._backpropagate(path, reward)
            # print("  Backpropagation complete.")

        end_time = time.time()
        print(f"Search completed {num_simulations} sims in {end_time - start_time:.2f} seconds.")

        # Choose the best move from the root node
        best_move = self._get_best_move(root)
        return best_move


    def _select_best_uct_child(self, node):
        """
        Selects the child with the highest UCT value (Upper Confidence Bound applied to Trees).
        Uses the negamax perspective: the value of a child node is considered from the parent's perspective.
        """
        # Parent visit count must be > 0 if we are selecting a child
        log_parent_visits = math.log(node.visit_count)

        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            if child.visit_count == 0:
                # If any child is unvisited, UCT score is infinite.
                # This case shouldn't ideally be reached if selection only happens on fully expanded nodes,
                # but handle defensively. In practice, expansion usually picks before this state.
                # However, if we *must* select from among visited children (e.g., if expansion logic changes),
                # an unvisited child has max priority. Let's assume expansion handles this.
                # A robust selection prioritizes unvisited, but here we assume visit_count > 0 for UCT calc.
                # If this function IS called when some children have 0 visits, it's a logic issue elsewhere.
                 # Let's assume children passed here have visit_count > 0 because the node is fully expanded
                 # and has been visited before. Add check:
                 if child.visit_count == 0:
                      print(f"Warning: Selecting from fully expanded node, but child {child.move} has 0 visits.")
                      # Assign infinite score to prioritize exploring it if it somehow got missed.
                      score = float('inf')
                 else:
                    # child.get_value_estimate() is the average outcome for the player *at the child node*.
                    # The parent node wants to maximize its *own* outcome. Since the child's player
                    # is the opponent, the parent maximizes the negative of the child's value estimate.
                    exploitation_term = -child.get_value_estimate()

                    exploration_term = self.C * math.sqrt(log_parent_visits / child.visit_count)

                    score = exploitation_term + exploration_term

            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
             # This should not happen if node.children is not empty.
             print(f"Error: No best child found during selection for node with state:\n{node.state}")
             # Fallback: return a random child
             best_child = random.choice(list(node.children.values()))

        return best_child


    def _expand(self, node):
        """
        Expands the node by creating a child node for one of its untried actions.
        Assumes the node is not terminal and not fully expanded.
        """
        action = node.untried_actions.pop() # Get and remove one untried action
        child_state = node.state.clone()
        child_state.apply_action(action)
        child_node = MCTSNode(child_state, parent=node, move=action)
        node.children[action] = child_node
        return child_node


    def _rollout(self, state):
        """
        Simulates a random playout from the given state until a terminal state is reached.
        Returns the terminal rewards as a list [P0_reward, P1_reward].
        """
        rollout_state = state.clone()
        while not rollout_state.is_terminal():
            legal_actions = rollout_state.legal_actions()
            if not legal_actions: # Should not happen in TicTacToe unless already terminal
                 break
            action = random.choice(legal_actions)
            rollout_state.apply_action(action)
        # Returns the utility for each player at the end of the episode.
        # For TicTacToe: +1 for win, -1 for loss, 0 for draw.
        return rollout_state.returns()


    def _backpropagate(self, path, rewards):
        """
        Updates the visit counts and total values of nodes along the search path.
        The value update considers the perspective of the player whose turn it was at each node.
        `rewards` is the list [P0_reward, P1_reward] from the rollout.
        """
        # Iterate backwards from the leaf node up to the root's parent
        for node in reversed(path):
            node.visit_count += 1

            # Determine the reward relevant to the player whose turn it was at this node.
            player = node.player_to_move
            if player == 0:
                reward_for_player = rewards[0] # Use Player 0's outcome
            elif player == 1:
                reward_for_player = rewards[1] # Use Player 1's outcome
            else:
                 # This might happen for the terminal node added during expansion,
                 # or if root was terminal. Player should be defined.
                 # If it's terminal, its value doesn't influence parent selection directly,
                 # but its visit count matters. Let's assign reward based on parent.
                 if node.parent:
                      parent_player = node.parent.player_to_move
                      # Reward should be from opponent's perspective relative to parent
                      reward_for_player = rewards[1-parent_player]
                 else: # Root is terminal - shouldn't happen due to initial check
                      reward_for_player = 0 # Or handle error
                      print("Warning: Backpropagating on a terminal root or node with undefined player.")


            # Add this reward to the node's total value.
            # total_value accumulates rewards from the perspective of node.player_to_move.
            node.total_value += reward_for_player


    def _get_best_move(self, root_node):
        """
        Selects the best move from the root node after the search.
        Typically chooses the move leading to the most visited child node for robustness.
        """
        if not root_node.children:
            print("Warning: Root node has no children after search. Choosing random.")
            return random.choice(root_node.legal_actions)

        most_visited_child = None
        max_visits = -1

        # Debug print: Show stats for root's children
        # print("\nRoot Children Stats:")
        # sorted_children = sorted(root_node.children.items(), key=lambda item: item[1].visit_count, reverse=True)
        # for move, child in sorted_children:
        #      child_value = child.get_value_estimate()
        #      # Value from root's perspective = -child_value
        #      print(f"  Move: {move}, Visits: {child.visit_count}, Child_Val (P{child.player_to_move}'s persp): {child_value:.4f}, Root_Persp_Val: {-child_value:.4f}")


        for move, child in root_node.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                most_visited_child = child

        if most_visited_child is None:
             print("Error: Could not find most visited child. Choosing random.")
             return random.choice(root_node.legal_actions)

        return most_visited_child.move


# --- Game Playing Logic ---

def play_game(game_name="tic_tac_toe", num_simulations=10000, exploration_c=math.sqrt(2)):
    """Plays a full game using MCTS for both players."""
    game = pyspiel.load_game(game_name)
    state = game.new_initial_state()
    mcts_agent = MCTS(exploration_constant=exploration_c)

    print(f"Starting {game.get_type().long_name} Game")
    print(f"MCTS settings: Sims={num_simulations}, C={exploration_c:.3f}")

    turn = 0
    while not state.is_terminal():
        current_player = state.current_player()
        print("-" * 20)
        print(f"Turn {turn}, Player {current_player} to move:")
        print(state)

        start_search_time = time.time()
        print(f"Player {current_player} thinking...")
        best_move = mcts_agent.search(state, num_simulations)
        end_search_time = time.time()
        print(f"Search time: {end_search_time - start_search_time:.2f} seconds.")


        # Validate move (sanity check)
        if best_move not in state.legal_actions():
             print(f"\n!!! ERROR: MCTS chose illegal move {best_move} !!!")
             print(f"Legal moves: {state.legal_actions()}")
             # Try to recover by choosing most visited among legal, or random
             try:
                 # This requires access to the root node internal to the search, which we don't have here easily.
                 # Fallback to random legal move.
                 print("Choosing random legal move as fallback.")
                 best_move = random.choice(state.legal_actions())
             except IndexError:
                 print("No legal moves available? State should be terminal.")
                 break # Exit loop

        print(f"Player {current_player} selects move: {best_move}\n")
        state.apply_action(best_move)
        turn += 1

    # Game finished
    print("=" * 20)
    print("Game Over!")
    print("Final state:")
    print(state)
    returns = state.returns()
    print(f"Returns: Player 0: {returns[0]}, Player 1: {returns[1]}")
    if returns[0] > returns[1]:
        print("Player 0 Wins!")
    elif returns[1] > returns[0]:
        print("Player 1 Wins!")
    else:
        print("It's a Draw!")

if __name__ == "__main__":
    # Use a high number of simulations for TicTacToe - it should play perfectly.
    # 10k should be sufficient, 20k+ very safe. Let's use 10k default.
    play_game(num_simulations=10000)
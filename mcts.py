import math
import random

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state    
        self.parent = parent  
        self.move = move      
        self.children = {}    
        self.visit_count = 0
        self.total_value = 0.0

    def is_terminal(self, game):
        return game.is_terminal(self.state)

    def is_fully_expanded(self, game):
        if self.is_terminal(game):
            return True
        legal_moves = game.get_legal_moves(self.state)
        return len(self.children) == len(legal_moves)

    def value(self):
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

class MCTS:
    def __init__(self, game, exploration_weight=1.0, gamma=1.0):
        """
        game: a game object that implements:
              - get_legal_moves(state)
              - next_state(state, move)
              - is_terminal(state)
              - get_reward(state)
        exploration_weight: controls the trade-off between exploration and exploitation
        gamma: discount factor (set to 1.0 for undiscounted rollouts)
        """
        self.game = game
        self.exploration_weight = exploration_weight
        self.gamma = gamma

    def search(self, initial_state, num_simulations):
        root = Node(initial_state)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # 1. Selection: Traverse the tree until reaching a node that is not fully expanded or is terminal.
            while node.is_fully_expanded(self.game) and not node.is_terminal(self.game):
                move, node = self._select_child(node)
                search_path.append(node)

            # 2. Expansion: If the node is non-terminal, expand one of its unvisited children.
            if not node.is_terminal(self.game):
                legal_moves = self.game.get_legal_moves(node.state)
                untried_moves = [m for m in legal_moves if m not in node.children]
                if untried_moves:
                    move = random.choice(untried_moves)
                    next_state = self.game.next_state(node.state, move)
                    child_node = Node(next_state, parent=node, move=move)
                    node.children[move] = child_node
                    node = child_node
                    search_path.append(node)

            # 3. Simulation (Rollout): Perform a full random rollout from the new node until a terminal state is reached.
            reward = self._rollout(node.state)

            # 4. Backpropagation: Update the values and visit counts of nodes along the search path.
            self._backpropagate(search_path, reward)

        # Choose the best move based on the highest average value.
        best_move = None
        best_value = -float('inf')
        for move, child in root.children.items():
            child_value = child.value()
            if child_value > best_value:
                best_value = child_value
                best_move = move

        return best_move, best_value

    def _select_child(self, node):
        """Select a child node using the UCT (Upper Confidence Bound for Trees) formula."""
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in node.children.items():
            # UCT formula: Q / N + exploration_weight * sqrt(ln(total_visits) / N)
            exploitation = child.value()
            exploration = self.exploration_weight * math.sqrt(math.log(total_visits) / child.visit_count)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _rollout(self, state):
        """
        Perform a random rollout from the given state until a terminal state is reached.
        The rollout returns the reward from the terminal state.
        """
        current_state = state
        cumulative_reward = 0.0
        discount = 1.0

        while not self.game.is_terminal(current_state):
            legal_moves = self.game.get_legal_moves(current_state)
            if not legal_moves:  
                break
            move = random.choice(legal_moves)
            current_state = self.game.next_state(current_state, move)
            # Assume the game provides a step reward if needed. If not, reward is only given at terminal state.
            reward = self.game.get_reward(current_state)
            cumulative_reward += discount * reward
            discount *= self.gamma

        return cumulative_reward

    def _backpropagate(self, search_path, reward):
        """
        Propagate the reward back through the path.
        Each node updates its visit count and total value.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += reward
            reward *= self.gamma  

class SimpleGame:
    def __init__(self, target_value=99):
        self.target_value = target_value

    def get_legal_moves(self, state):
        if state < self.target_value:
            return [1, 2, 3]  # Example moves: add 1, 2, or 3
        else:
            return []

    def next_state(self, state, move):
        return state + move

    def is_terminal(self, state):
        return state >= self.target_value

    def get_reward(self, state):
        if self.is_terminal(state):
            return 1.0 if state == self.target_value else -5.0 
        return -0.5 

# Test case
def test_mcts():
    game = SimpleGame()
    mcts = MCTS(game)
    initial_state = 0
    best_move, best_value = mcts.search(initial_state, num_simulations=1000)

    print(f"Best move: {best_move}")
    print(f"Best value: {best_value}")

    next_state = game.next_state(initial_state, best_move)
    print(f"Next state: {next_state}")
    assert next_state > initial_state

test_mcts()
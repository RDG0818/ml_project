import matplotlib.pyplot as plt
import numpy as np
import re

# Read and parse the data from the file
algorithms = []
current_algorithm = None

with open('checkers_data.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        # Detect algorithm section headers
        if line.startswith('MCTS:') or line.startswith('Multiprocessing MCTS:') or \
           line.startswith('Tree Parallel MCTS:') or line.startswith('Speculative MCTS:'):
            current_algorithm = line.split(':')[0].strip()
            algorithms.append({
                'name': current_algorithm,
                'times': [],
                'total': None,
                'average': None
            })
        else:
            # Extract game times
            game_time_match = re.match(r'^Game \d+.* (\d+\.\d+) seconds', line)
            if game_time_match:
                time = float(game_time_match.group(1))
                algorithms[-1]['times'].append(time)
            else:
                # Extract total time
                total_time_match = re.match(r'Total Time for 20 games: (\d+\.\d+) seconds', line)
                if total_time_match:
                    algorithms[-1]['total'] = float(total_time_match.group(1))
                else:
                    # Extract average time (if present)
                    avg_time_match = re.match(r'Average Time for 20 games: (\d+\.\d+) seconds', line)
                    if avg_time_match:
                        algorithms[-1]['average'] = float(avg_time_match.group(1))

# Prepare data for plotting
algorithm_names = [algo['name'] for algo in algorithms]
total_times = [algo['total'] for algo in algorithms]
times_data = [algo['times'] for algo in algorithms]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})

# Bar chart for total times
bars = ax1.bar(algorithm_names, total_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_ylabel('Total Time (seconds)', fontsize=12)
ax1.set_title('Total Time for 20 Games of Checkers', fontsize=14, pad=20)
ax1.tick_params(axis='x', labelsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add total time values on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom',
                 fontsize=10)

# Boxplot for individual game times
ax2.boxplot(times_data, labels=algorithm_names, patch_artist=True,
            boxprops=dict(facecolor='lightgray', color='gray'),
            medianprops=dict(color='black'))
ax2.set_ylabel('Game Time (seconds)', fontsize=12)
ax2.set_title('Distribution of Individual Game Times (Boxplot)', fontsize=14, pad=20)
ax2.tick_params(axis='x', labelsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
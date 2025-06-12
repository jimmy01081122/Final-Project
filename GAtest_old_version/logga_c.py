
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, Markdown
import numpy as np
import tsplib95
import random
import time
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import json
import os
plt.rcParams['animation.embed_limit'] = 2**128

# Global variables
INF = 9999999
dist_matrix = []
use_numpy = 0  # 0 = do not use numpy arrays for individuals, 1 = use numpy arrays

class ParameterTuner:
    """
    Handles automatic parameter tuning for the TSP GA solver.
    """
    
    def __init__(self, target_cost, max_attempts=50, improvement_threshold=0.95):
        """
        :param target_cost: Target cost to achieve
        :param max_attempts: Maximum number of tuning attempts
        :param improvement_threshold: Minimum improvement ratio to continue tuning
        """
        self.target_cost = target_cost
        self.max_attempts = max_attempts
        self.improvement_threshold = improvement_threshold
        
        # Parameter ranges for tuning
        self.param_ranges = {
            'population_size': [50, 100, 200, 300, 500],
            'max_gens': [500, 1000, 1500, 2000, 3000],
            'cx_prob': [0.7, 0.8, 0.9, 0.95],
            'mut_prob': [0.05, 0.1, 0.15, 0.2, 0.3],
            'tournament_size': [2, 3, 4, 5, 6, 8]
        }
        
        # Starting parameters
        self.current_params = {
            'population_size': 100,
            'max_gens': 1000,
            'cx_prob': 0.9,
            'mut_prob': 0.1,
            'tournament_size': 4
        }
        
        self.best_cost = float('inf')
        self.best_params = self.current_params.copy()
        self.attempt_history = []
        
    def adjust_parameters(self, current_cost, attempt_num):
        """
        Adjust parameters based on current performance.
        """
        print(f"\nAttempt {attempt_num}: Current cost = {current_cost}, Target = {self.target_cost}")
        
        # Record this attempt
        self.attempt_history.append({
            'attempt': attempt_num,
            'params': self.current_params.copy(),
            'cost': current_cost
        })
        
        # Update best if improved
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_params = self.current_params.copy()
            print(f"New best cost: {self.best_cost}")
        
        # If target reached, we're done
        if current_cost <= self.target_cost:
            print(f"🎉 Target cost {self.target_cost} achieved with cost {current_cost}!")
            return True
        
        # Adaptive parameter adjustment strategy
        improvement_ratio = self.best_cost / current_cost if current_cost > 0 else 0
        
        if improvement_ratio < self.improvement_threshold and attempt_num > 5:
            # If not improving much, try more aggressive changes
            print("Low improvement - applying aggressive parameter changes")
            self._aggressive_parameter_change()
        else:
            # Normal parameter adjustment
            self._normal_parameter_adjustment(current_cost, attempt_num)
        
        print(f"Adjusted parameters: {self.current_params}")
        return False
    
    def _normal_parameter_adjustment(self, current_cost, attempt_num):
        """
        Normal parameter adjustment based on performance patterns.
        """
        # If cost is much higher than target, increase exploration
        cost_ratio = current_cost / self.target_cost if self.target_cost > 0 else 1
        
        if cost_ratio > 1.5:  # Cost is much higher than target
            # Increase population and generations for better exploration
            if self.current_params['population_size'] < max(self.param_ranges['population_size']):
                self.current_params['population_size'] = min(
                    self.current_params['population_size'] * 1.5, 
                    max(self.param_ranges['population_size'])
                )
            
            if self.current_params['max_gens'] < max(self.param_ranges['max_gens']):
                self.current_params['max_gens'] = min(
                    int(self.current_params['max_gens'] * 1.2), 
                    max(self.param_ranges['max_gens'])
                )
            
            # Increase mutation for more exploration
            if self.current_params['mut_prob'] < max(self.param_ranges['mut_prob']):
                self.current_params['mut_prob'] = min(
                    self.current_params['mut_prob'] * 1.2, 
                    max(self.param_ranges['mut_prob'])
                )
        
        elif cost_ratio > 1.1:  # Close to target but not quite there
            # Fine-tune parameters
            if attempt_num % 3 == 0:
                # Try different crossover probability
                idx = self.param_ranges['cx_prob'].index(self.current_params['cx_prob'])
                if idx < len(self.param_ranges['cx_prob']) - 1:
                    self.current_params['cx_prob'] = self.param_ranges['cx_prob'][idx + 1]
            
            if attempt_num % 4 == 0:
                # Try different tournament size
                if self.current_params['tournament_size'] < max(self.param_ranges['tournament_size']):
                    self.current_params['tournament_size'] += 1
    
    def _aggressive_parameter_change(self):
        """
        Apply more aggressive parameter changes when stuck.
        """
        # Randomly select new parameters from ranges
        for param, ranges in self.param_ranges.items():
            if random.random() < 0.6:  # 60% chance to change each parameter
                self.current_params[param] = random.choice(ranges)
        
        # Ensure population size and generations are large enough for difficult problems
        self.current_params['population_size'] = max(self.current_params['population_size'], 200)
        self.current_params['max_gens'] = max(self.current_params['max_gens'], 1500)
    
    def save_tuning_history(self, filename="tuning_history.json"):
        """
        Save tuning history to file.
        """
        history_data = {
            'best_params': self.best_params,
            'best_cost': self.best_cost,
            'target_cost': self.target_cost,
            'attempt_history': self.attempt_history
        }
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Tuning history saved to {filename}")


class AnimationTSP:
    """
    This class creates an animated visualization of a TSP solution improving over time.
    """

    def __init__(self, history, x_coords, y_coords, costs):
        """
        :param history: A list of solutions (each solution is a list of city indices).
        :param x_coords: List of x-coordinates of each city.
        :param y_coords: List of y-coordinates of each city.
        :param costs: A list of costs corresponding to each solution in history.
        """

        # Ensure 'history' is a list of lists
        if isinstance(history[0], list):
            self.history = history
        else:
            self.history = [h.tolist() for h in history]

        self.costs = costs
        self.points = np.column_stack((x_coords, y_coords))

        self.fig, self.ax = plt.subplots()
        self.line, = plt.plot([], [], lw=2)
        self.title = self.ax.text(
            0.8, 1.035, "", 
            bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
            transform=self.ax.transAxes,
            ha="center"
        )

    def init_animation(self):
        """
        Initializes the animation with empty lines and plots the city nodes.
        """
        # Plot the city nodes from the first solution in the history
        x_plot = [self.points[i][0] for i in self.history[0]]
        y_plot = [self.points[i][1] for i in self.history[0]]
        plt.plot(x_plot, y_plot, 'co')

        # Adjust axes with a small margin
        extra_x = (max(x_plot) - min(x_plot)) * 0.05
        extra_y = (max(y_plot) - min(y_plot)) * 0.05
        self.ax.set_xlim(min(x_plot) - extra_x, max(x_plot) + extra_x)
        self.ax.set_ylim(min(y_plot) - extra_y, max(y_plot) + extra_y)

        # Initialize the route line as empty
        self.line.set_data([], [])
        return self.line,

    def update_animation(self, frame):
        """
        For each frame, update the plot with the route of that generation/iteration.
        """
        route = self.history[frame]
        x_plot = [self.points[i, 0] for i in route + [route[0]]]
        y_plot = [self.points[i, 1] for i in route + [route[0]]]

        self.title.set_text(f"Iteration {frame}, Cost {self.costs[frame]}")
        self.line.set_data(x_plot, y_plot)
        return self.line

    def animate_routes(self):
        """
        Creates and displays the animation in a Jupyter environment.
        """
        # Setting how many frames to skip to create a shorter animation
        div = len(self.history) // 3 if len(self.history) > 3 else 1
        step = len(self.history) // div if div != 0 else 1

        ani = FuncAnimation(
            self.fig, 
            self.update_animation, 
            frames=range(0, len(self.history), step),
            init_func=self.init_animation, 
            interval=3,
            repeat=False
        )

        plt.title("TSP Route Animation")

        # Convert animation to HTML for display
        ani.interactive = True 
        html_anim = ani.to_jshtml()
        display(HTML(html_anim))


class TSPInstance:
    """
    Loads a TSP instance via tsplib95 and prepares data (coordinates, distance matrix).
    """

    def __init__(self, plot_route, instance_file):
        """
        :param plot_route: Boolean (0 or 1) to indicate whether to plot/animate the route.
        :param instance_file: File path to the TSP instance (TSPLIB format).
        """
        self.plot_route = bool(plot_route)
        self.plot_enabled = self.plot_route

        self.coord_x = []
        self.coord_y = []
        self.problem = tsplib95.load(instance_file)
        self.info = self.problem.as_keyword_dict()
        self.n = len(self.problem.get_graph())

        # If the instance can be plotted (EUC_2D, GEO, ATT), save city coordinates
        if self.plot_route and self._can_plot():
            for i in range(1, self.n + 1):
                x, y = self.info['NODE_COORD_SECTION'][i]
                self.coord_x.append(x)
                self.coord_y.append(y)
        else:
            self.plot_route = False

    def _can_plot(self):
        """
        Checks if the TSP instance has a coordinate-based distance (e.g., EUC_2D, GEO, ATT).
        """
        dist_type = self.info['EDGE_WEIGHT_TYPE']
        if dist_type in ['EUC_2D', 'GEO', 'ATT']:
            return True
        else:
            print("Plotting is not supported for this EDGE_WEIGHT_TYPE.")
            return False

    def generate_distance_matrix(self):
        """
        Generate the global distance matrix (dist_matrix) for the TSP.
        """
        global dist_matrix
        dist_matrix = [[INF for _ in range(self.n)] for _ in range(self.n)]
        start_node = list(self.problem.get_nodes())[0]

        # Adjust if the node indices start at 1 or 0
        if start_node == 0:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        dist_matrix[i][j] = self.problem.get_weight(i, j)
        else:
            # If nodes start at 1 instead of 0
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        dist_matrix[i][j] = self.problem.get_weight(i + 1, j + 1)


def distance(i, j):
    """
    Returns the distance between city i and city j using the global distance matrix.
    """
    return dist_matrix[i][j]


def total_cost(route):
    """
    Evaluates the total cost of a TSP route (closed tour).
    """
    csum = 0
    for k in range(len(route) - 1):
        csum += distance(route[k], route[k + 1])
    # Add cost from last city back to the first city
    csum += distance(route[-1], route[0])
    return (csum,)


def nearest_neighbor(n):
    """
    Generates a TSP route using the Nearest Neighbor heuristic with probability 0.4,
    otherwise shuffles cities randomly.

    :param n: Number of cities.
    :return: A route (list or numpy array) representing a permutation of cities.
    """
    start = random.randrange(0, n)
    if random.uniform(0, 1) < 0.4:
        current = start
        route = [start]
        selected = [False] * n
        selected[current] = True

        while len(route) < n:
            min_dist = INF
            next_city = None
            for candidate in range(n):
                if not selected[candidate] and candidate != current:
                    cost_val = distance(current, candidate)
                    if cost_val < min_dist:
                        min_dist = cost_val
                        next_city = candidate

            route.append(next_city)
            selected[next_city] = True
            current = next_city
    else:
        route = list(range(n))
        random.shuffle(route)

    if use_numpy:
        return np.array(route)
    else:
        return route


def two_opt(route):
    """
    2-Opt local search: tries to improve the route by reversing segments.
    """
    n = len(route)
    improved = False
    best_delta = 0
    cut_count = 0

    # Shift the route starting point randomly
    k = random.randint(0, n - 1)
    if use_numpy:
        route = np.hstack((route[k:], route[:k]))  # rotate with numpy
    else:
        route = route[k:] + route[:k]

    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            old_cost = distance(route[i], route[i + 1]) + distance(route[j], route[j + 1])
            new_cost = distance(route[i], route[j]) + distance(route[i + 1], route[j + 1])
            delta = new_cost - old_cost

            if delta < best_delta:
                best_delta = delta
                min_i, min_j = i, j
                cut_count += 1
                # Only make one improving swap
                if cut_count == 1:
                    improved = True

        if improved:
            break

    if cut_count > 0:
        segment = route[min_i + 1: min_j + 1]
        route[min_i + 1: min_j + 1] = segment[::-1]


def perturbation_swap_two(route):
    """
    Perturbation: randomly swap two distinct cities in the route.
    """
    i, j = 0, 0
    n = len(route)
    while i == j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

    route[i], route[j] = route[j], route[i]


def perturbation_swap_neighbors(route):
    """
    Perturbation: choose one city at random and swap it with its immediate neighbor.
    """
    n = len(route)
    i = random.randint(0, n - 1)
    j = i + 1 if i < n - 1 else 0

    route[i], route[j] = route[j], route[i]


def perturbation_reverse_subroute(route):
    """
    Perturbation 2: choose two random points i, j (i < j) and reverse the subroute between them.
    """
    i, j = 0, 0
    n = len(route)
    while i >= j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

    route[i:j] = route[i:j][::-1]


def mutate(route):
    """
    Custom mutation strategy that applies a subroute reversal (2).
    Other perturbations are commented out but can be included if desired.
    """
    # Examples of different perturbations:
    # perturbation_swap_two(route)
    # perturbation_swap_neighbors(route)
    # two_opt(route)
    perturbation_reverse_subroute(route)
    return (route,)


def ga_tunable(tsp_instance, params, seed, verbose=True):
    """
    Tunable GA to solve TSP with configurable parameters.
    :param tsp_instance: TSPInstance object.
    :param params: Dictionary containing GA parameters.
    :param seed: Random seed to control reproducibility.
    :param verbose: Whether to print progress information.
    :return: Best cost achieved.
    """
    population_size = int(params['population_size'])
    max_gens = int(params['max_gens'])
    cx_prob = params['cx_prob']
    mut_prob = params['mut_prob']
    tournament_size = int(params['tournament_size'])
    n_cities = tsp_instance.n

    random.seed(seed)

    # Clear any existing creators to avoid conflicts
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    # Define Fitness and Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if use_numpy:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    else:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("indices", nearest_neighbor, n_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", total_cost)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", mutate)

    # Build initial population
    pop = toolbox.population(n=population_size)
    if use_numpy:
        hof = tools.HallOfFame(1, similar=np.array_equal)
    else:
        hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Evaluate initial population
    start_time = time.time()
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit_val in zip(pop, fitnesses):
        ind.fitness.values = fit_val

    gen = 0
    solutions_history = []
    costs_history = []

    record = stats.compile(pop)
    logbook.record(gen=gen, evals=len(pop), **record)
    if verbose:
        print(f"Gen {logbook[-1]['gen']}: Avg={logbook[-1]['avg']:.1f}, Min={logbook[-1]['min']:.1f}")

    # Evolve
    while gen < max_gens:
        gen += 1

        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate mutated/crossover offsprings
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_inds)
        for ind, fit_val in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit_val

        # Replace population
        pop[:] = offspring
        hof.update(offspring)

        record = stats.compile(offspring)
        logbook.record(gen=gen, evals=len(offspring), **record)
        
        if verbose and gen % 100 == 0:
            print(f"Gen {logbook[-1]['gen']}: Avg={logbook[-1]['avg']:.1f}, Min={logbook[-1]['min']:.1f}")

        # Store best route of this generation
        best_ind = tools.selBest(offspring, k=1)[0]
        solutions_history.append(best_ind)
        costs_history.append(int(logbook[-1]["min"]))

    end_time = time.time()
    
    best_cost = min(costs_history)
    
    if verbose:
        print(f"Final best cost: {best_cost}")
        print(f"Execution time: {end_time - start_time:.2f}s")

    # If route plotting is enabled, animate the route
    if tsp_instance.plot_route and verbose:
        anim = AnimationTSP(solutions_history, tsp_instance.coord_x, tsp_instance.coord_y, costs_history)
        anim.animate_routes()

    return best_cost


def self_tuning_tsp_solver(instance_file, target_cost, max_attempts=50):
    """
    Main self-tuning TSP solver function.
    
    :param instance_file: Path to TSP instance file
    :param target_cost: Target cost to achieve
    :param max_attempts: Maximum tuning attempts
    """
    print("="*60)
    print("  🤖 Self-Tuning TSP GA Solver")
    print("="*60)
    print(f"Instance: {instance_file}")
    print(f"Target cost: {target_cost}")
    print(f"Max attempts: {max_attempts}")
    print("="*60)
    
    global use_numpy
    use_numpy = 0  # Use lists for simplicity
    
    # Create TSP instance
    tsp_instance = TSPInstance(plot_route=0, instance_file=instance_file)  # Disable plotting for faster execution
    tsp_instance.generate_distance_matrix()
    
    print(f"Loaded TSP instance with {tsp_instance.n} cities")
    
    # Initialize parameter tuner
    tuner = ParameterTuner(target_cost, max_attempts)
    
    best_overall_cost = float('inf')
    best_overall_params = None
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*20} ATTEMPT {attempt} {'='*20}")
        
        # Use different seeds for each attempt to explore different solutions
        seed = 42 + attempt
        
        # Run GA with current parameters
        try:
            current_cost = ga_tunable(
                tsp_instance, 
                tuner.current_params, 
                seed, 
                verbose=(attempt <= 5 or attempt % 5 == 0)  # Reduce verbosity after first few attempts
            )
            
            # Update overall best
            if current_cost < best_overall_cost:
                best_overall_cost = current_cost
                best_overall_params = tuner.current_params.copy()
            
            # Check if we should stop (target reached or adjust parameters)
            if tuner.adjust_parameters(current_cost, attempt):
                break
                
        except Exception as e:
            print(f"Error in attempt {attempt}: {e}")
            tuner._aggressive_parameter_change()  # Try different parameters
            continue
    
    # Final results
    print("\n" + "="*60)
    print("  🎯 SELF-TUNING RESULTS")
    print("="*60)
    print(f"Best cost achieved: {best_overall_cost}")
    print(f"Target cost: {target_cost}")
    print(f"Target achieved: {'✅ YES' if best_overall_cost <= target_cost else '❌ NO'}")
    print(f"Total attempts: {len(tuner.attempt_history)}")
    print(f"Best parameters: {best_overall_params}")
    
    # Save results
    tuner.save_tuning_history()
    
    return best_overall_cost, best_overall_params


def plot_evolution(min_values, avg_values):
    """
    Plots the evolution of the best and average cost over generations.
    """
    plt.figure()
    plot1, = plt.plot(min_values, 'c-', label='Best Cost')
    plot2, = plt.plot(avg_values, 'b-', label='Average Cost')
    plt.legend(handles=[plot1, plot2], frameon=True)
    plt.ylabel('Cost')
    plt.xlabel('Generations')
    plt.title("Generations vs. Cost - TSP")
    plt.xlim((0, len(min_values)))
    plt.show()


def main():
    """
    Main function for self-tuning TSP solver.
    """
    print("=========================================")
    print("  🤖 Self-Tuning TSP GA Solver")
    print("=========================================")
    
    # Example usage
    instance_file = input("Enter TSP instance file path (e.g., kroA100.tsp): ")
    target_cost = float(input("Enter target cost to achieve: "))
    max_attempts = int(input("Enter maximum tuning attempts (default 30): ") or "30")
    
    # Run self-tuning solver
    best_cost, best_params = self_tuning_tsp_solver(instance_file, target_cost, max_attempts)
    
    print(f"\n🏁 Self-tuning completed!")
    print(f"Best cost: {best_cost}")
    print(f"Best parameters: {best_params}")


if __name__ == "__main__":
    main()

# best_cost, best_params = self_tuning_tsp_solver("kroA100.tsp", 21500, 50)
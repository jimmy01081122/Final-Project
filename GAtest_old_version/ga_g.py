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
plt.rcParams['animation.embed_limit'] = 2**128

# Global variables
INF = 9999999
dist_matrix = []
use_numpy = 0  # 0 = do not use numpy arrays for individuals, 1 = use numpy arrays

class AnimationTSP:
    """
    This class creates an animated visualization of a TSP solution improving over time.
    """
    def __init__(self, history, x_coords, y_coords, costs):
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
        x_plot = [self.points[i][0] for i in self.history[0]]
        y_plot = [self.points[i][1] for i in self.history[0]]
        plt.plot(x_plot, y_plot, 'co')
        extra_x = (max(x_plot) - min(x_plot)) * 0.05
        extra_y = (max(y_plot) - min(y_plot)) * 0.05
        self.ax.set_xlim(min(x_plot) - extra_x, max(x_plot) + extra_x)
        self.ax.set_ylim(min(y_plot) - extra_y, max(y_plot) + extra_y)
        self.line.set_data([], [])
        return self.line,

    def update_animation(self, frame):
        route = self.history[frame]
        x_plot = [self.points[i, 0] for i in route + [route[0]]]
        y_plot = [self.points[i, 1] for i in route + [route[0]]]
        self.title.set_text(f"Iteration {frame}, Cost {self.costs[frame]}")
        self.line.set_data(x_plot, y_plot)
        return self.line

    def animate_routes(self):
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
        ani.interactive = True 
        html_anim = ani.to_jshtml()
        display(HTML(html_anim))

class TSPInstance:
    """
    Loads a TSP instance via tsplib95 and prepares data (coordinates, distance matrix).
    """
    def __init__(self, plot_route, instance_file):
        self.plot_route = bool(plot_route)
        self.plot_enabled = self.plot_route
        self.coord_x = []
        self.coord_y = []
        self.problem = tsplib95.load(instance_file)
        self.info = self.problem.as_keyword_dict()
        self.n = len(self.problem.get_graph())
        if self.plot_route and self._can_plot():
            for i in range(1, self.n + 1):
                x, y = self.info['NODE_COORD_SECTION'][i]
                self.coord_x.append(x)
                self.coord_y.append(y)
        else:
            self.plot_route = False

    def _can_plot(self):
        dist_type = self.info['EDGE_WEIGHT_TYPE']
        if dist_type in ['EUC_2D', 'GEO', 'ATT']:
            return True
        else:
            print("Plotting is not supported for this EDGE_WEIGHT_TYPE.")
            return False

    def generate_distance_matrix(self):
        global dist_matrix
        dist_matrix = [[INF for _ in range(self.n)] for _ in range(self.n)]
        start_node = list(self.problem.get_nodes())[0]
        if start_node == 0:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        dist_matrix[i][j] = self.problem.get_weight(i, j)
        else:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        dist_matrix[i][j] = self.problem.get_weight(i + 1, j + 1)

def distance(i, j):
    return dist_matrix[i][j]

def total_cost(route):
    csum = 0
    for k in range(len(route) - 1):
        csum += distance(route[k], route[k + 1])
    csum += distance(route[-1], route[0])
    return (csum,)

def nearest_neighbor(n):
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
    n = len(route)
    improved = False
    best_delta = 0
    cut_count = 0
    k = random.randint(0, n - 1)
    if use_numpy:
        route = np.hstack((route[k:], route[:k]))
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
                if cut_count == 1:
                    improved = True
        if improved:
            break
    if cut_count > 0:
        segment = route[min_i + 1: min_j + 1]
        route[min_i + 1: min_j + 1] = segment[::-1]

def perturbation_swap_two(route):
    i, j = 0, 0
    n = len(route)
    while i == j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
    route[i], route[j] = route[j], route[i]

def perturbation_swap_neighbors(route):
    n = len(route)
    i = random.randint(0, n - 1)
    j = i + 1 if i < n - 1 else 0
    route[i], route[j] = route[j], route[i]

def perturbation_reverse_subroute(route):
    i, j = 0, 0
    n = len(route)
    while i >= j:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
    route[i:j] = route[i:j][::-1]

def mutate(route):
    perturbation_reverse_subroute(route)
    return (route,)

def plot_evolution(min_values, avg_values):
    plt.figure()
    plot1, = plt.plot(min_values, 'c-', label='Best Cost')
    plot2, = plt.plot(avg_values, 'b-', label='Average Cost')
    plt.legend(handles=[plot1, plot2], frameon=True)
    plt.ylabel('Cost')
    plt.xlabel('Generations')
    plt.title("Generations vs. Cost - TSP")
    plt.xlim((0, len(min_values)))
    plt.show()

def run_ga(tsp_instance, population_size, max_gens, cx_prob, mut_prob, tournament_size, seed):
    n_cities = tsp_instance.n
    random.seed(seed)
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
    start_time = time.time()
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit_val in zip(pop, fitnesses):
        ind.fitness.values = fit_val
    gen = 0
    solutions_history = []
    costs_history = []
    record = stats.compile(pop)
    logbook.record(gen=gen, evals=len(pop), **record)
    print(logbook[-1]["gen"], logbook[-1]["avg"], logbook[-1]["min"])
    while gen < max_gens:
        gen += 1
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_inds)
        for ind, fit_val in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit_val
        pop[:] = offspring
        hof.update(offspring)
        record = stats.compile(offspring)
        logbook.record(gen=gen, evals=len(offspring), **record)
        print(logbook[-1]["gen"], logbook[-1]["avg"], logbook[-1]["min"])
        best_ind = tools.selBest(offspring, k=1)[0]
        solutions_history.append(best_ind)
        costs_history.append(int(logbook[-1]["min"]))
    end_time = time.time()
    print(f"Best route cost: {min(costs_history)}")
    print(f"Execution time: {end_time - start_time}")
    if tsp_instance.plot_route:
        anim = AnimationTSP(solutions_history, tsp_instance.coord_x, tsp_instance.coord_y, costs_history)
        anim.animate_routes()
    if tsp_instance.plot_enabled:
        min_values, avg_values = logbook.select("min", "avg")
        plot_evolution(min_values, avg_values)
    return hof[0].fitness.values[0], hof[0]

def main():
    print("=========================================")
    print("  TSP GA Solver - Interactive Menu")
    print("=========================================")
    plot_flag = int(input("Enable route plotting/animation? (0=No, 1=Yes): "))
    instance_file = input("Enter the file path of the TSP instance (e.g., kroA200.tsp): ")
    target_cost = float(input("Enter the target cost: "))
    seed_value = int(input("Enter random seed (integer), e.g. 42: "))
    global use_numpy
    use_numpy = int(input("Use numpy arrays for individuals? (0=No, 1=Yes): "))
    tsp_instance = TSPInstance(plot_flag, instance_file)
    tsp_instance.generate_distance_matrix()
    population_size = 100
    max_gens = 1000
    cx_prob = 0.9
    mut_prob = 0.1
    tournament_size = 4
    max_attempts = 5
    current_attempt = 0
    best_cost = float('inf')
    best_route = None
    while current_attempt < max_attempts:
        print(f"\nAttempt {current_attempt + 1} with parameters: population_size={population_size}, max_gens={max_gens}, cx_prob={cx_prob}, mut_prob={mut_prob}, tournament_size={tournament_size}")
        current_cost, current_route = run_ga(tsp_instance, population_size, max_gens, cx_prob, mut_prob, tournament_size, seed_value)
        if current_cost < best_cost:
            best_cost = current_cost
            best_route = current_route
        if best_cost <= target_cost:
            print(f"Target achieved! Best cost: {best_cost}")
            break
        else:
            current_attempt += 1
            print(f"Attempt {current_attempt}: Best cost {best_cost} exceeds target {target_cost}. Adjusting parameters...")
            population_size = min(int(population_size * 1.5), 500)
            max_gens = min(int(max_gens * 1.5), 2000)
            if mut_prob < 0.2:
                mut_prob += 0.05
    else:
        print(f"Target not achieved after {max_attempts} attempts. Best cost: {best_cost}")
    if tsp_instance.plot_route and best_route is not None:
        anim = AnimationTSP([best_route], tsp_instance.coord_x, tsp_instance.coord_y, [best_cost])
        anim.animate_routes()

if __name__ == "__main__":
    main()
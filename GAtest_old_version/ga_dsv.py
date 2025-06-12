import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import numpy as np
import tsplib95
import random
import time
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import sys
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


def ga_simple(tsp_instance, seed):
    """
    Simple GA to solve TSP.
    :param tsp_instance: TSPInstance object.
    :param seed: Random seed to control reproducibility.
    """
    population_size = 50
    max_gens = 200
    cx_prob = 0.9
    mut_prob = 0.4
    tournament_size = 4
    n_cities = tsp_instance.n

    random.seed(seed)

    # Create the Fitness and Individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if use_numpy:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    else:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Register functions
    toolbox.register("indices", nearest_neighbor, n_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", total_cost)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", mutate)

    # Generate initial population
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

    start_time = time.time()
    final_population, logbook = algorithms.eaSimple(
        pop, toolbox, cx_prob, mut_prob, max_gens,
        stats=stats, halloffame=hof
    )
    end_time = time.time()

    min_list, avg_list = logbook.select("min", "avg")

    print(f"Best route cost: {min(min_list)}")
    print(f"Execution time : {end_time - start_time}")

    if tsp_instance.plot_enabled:
        plot_evolution(min_list, avg_list)

def run_ga_advanced(tsp_instance, seed, params, plot_enable=False):
    """
    運行高級GA演算法，返回最佳成本、最佳路線和歷史數據
    
    :param tsp_instance: TSP實例
    :param seed: 隨機種子
    :param params: 包含GA參數的字典
    :param plot_enable: 是否記錄歷史數據用於繪圖
    :return: 包含結果的字典
    """
    # 從參數字典型取得參數值
    population_size = params['population_size']
    max_gens = params['max_gens']
    cx_prob = params['cx_prob']
    mut_prob = params['mut_prob']
    tournament_size = params['tournament_size']
    n_cities = tsp_instance.n

    random.seed(seed)

    # 清除之前的創建，避免重複定義錯誤
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual

    # 創建適應度和個體類
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if use_numpy:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    else:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # 註冊函數
    toolbox.register("indices", nearest_neighbor, n_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", total_cost)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", mutate)

    # 生成初始種群
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

    # 評估初始種群
    start_time = time.time()
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit_val in zip(pop, fitnesses):
        ind.fitness.values = fit_val

    gen = 0
    solutions_history = []
    costs_history = []
    min_values = []
    avg_values = []

    record = stats.compile(pop)
    logbook.record(gen=gen, evals=len(pop), **record)
    min_values.append(record['min'])
    avg_values.append(record['avg'])
    
    if plot_enable:
        best_ind = tools.selBest(pop, k=1)[0]
        solutions_history.append(best_ind)
        costs_history.append(int(record['min']))

    # 進化過程
    while gen < max_gens:
        gen += 1

        # 選擇
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 變異
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 重新評估變異/交叉後的後代
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_inds)
        for ind, fit_val in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit_val

        # 替換種群
        pop[:] = offspring
        hof.update(offspring)

        record = stats.compile(offspring)
        logbook.record(gen=gen, evals=len(offspring), **record)
        min_values.append(record['min'])
        avg_values.append(record['avg'])
        
        if plot_enable:
            best_ind = tools.selBest(offspring, k=1)[0]
            solutions_history.append(best_ind)
            costs_history.append(int(record['min']))

    end_time = time.time()
    
    best_route = hof[0]
    best_cost = best_route.fitness.values[0]
    
    return {
        'best_cost': best_cost,
        'best_route': best_route,
        'execution_time': end_time - start_time,
        'min_values': min_values,
        'avg_values': avg_values,
        'solutions_history': solutions_history if plot_enable else None,
        'costs_history': costs_history if plot_enable else None
    }

def auto_tune_ga(instance_file, target_cost, max_trials=10):
    """
    自動調整參數的GA流程，直到達到目標成本或超過最大嘗試次數
    
    :param instance_file: TSP實例文件路徑
    :param target_cost: 目標成本
    :param max_trials: 最大嘗試次數
    """
    # 創建TSP實例（禁用繪圖直到最後）
    tsp_instance = TSPInstance(plot_route=0, instance_file=instance_file)
    tsp_instance.generate_distance_matrix()
    
    # 初始參數設定
    params = {
        'population_size': 50,
        'max_gens': 500,
        'cx_prob': 0.85,
        'mut_prob': 0.15,
        'tournament_size': 4
    }
    
    # 參數調整策略
    adjustment_strategy = [
        {'population_size': 1.5},  # 增加種群大小
        {'max_gens': 1.5},         # 增加最大代數
        {'cx_prob': 0.95},         # 增加交叉概率
        {'mut_prob': 0.2},         # 增加變異概率
        {'tournament_size': 6},    # 增加錦標賽大小
        {'population_size': 2.0, 'max_gens': 2.0}  # 同時增加兩個參數
    ]
    
    best_result = None
    best_cost = float('inf')
    trial_results = []
    
    print(f"Starting auto-tuning for target cost: {target_cost}")
    print("===============================================")
    
    for trial in range(max_trials):
        print(f"\nTrial {trial+1}/{max_trials} with parameters:")
        print(f"  Population: {params['population_size']}, Generations: {params['max_gens']}")
        print(f"  Crossover: {params['cx_prob']}, Mutation: {params['mut_prob']}")
        print(f"  Tournament size: {params['tournament_size']}")
        
        # 運行GA（最後一次運行時啟用繪圖）
        plot_enable = (trial == max_trials - 1) or (trial == 0)
        result = run_ga_advanced(tsp_instance, trial, params, plot_enable)
        cost = result['best_cost']
        
        print(f"  Result: Cost = {cost:.2f}, Time = {result['execution_time']:.2f}s")
        
        # 保存結果
        trial_results.append({
            'params': params.copy(),
            'cost': cost,
            'time': result['execution_time']
        })
        
        # 檢查是否達到目標
        if cost <= target_cost:
            best_result = result
            best_cost = cost
            print(f"\n🎯 Target cost achieved in trial {trial+1}!")
            break
        
        # 更新最佳結果
        if cost < best_cost:
            best_result = result
            best_cost = cost
        
        # 調整參數 - 使用當前策略
        strategy = adjustment_strategy[trial % len(adjustment_strategy)]
        for key, value in strategy.items():
            if key in params:
                # 如果是數值型調整
                if isinstance(value, float):
                    params[key] = int(params[key] * value) if key in ['population_size', 'max_gens'] else params[key] * value
                # 如果是絕對值設定
                else:
                    params[key] = value
        
        # 參數邊界檢查
        params['population_size'] = max(20, min(params['population_size'], 500))
        params['max_gens'] = max(100, min(params['max_gens'], 2000))
        params['cx_prob'] = max(0.7, min(params['cx_prob'], 0.99))
        params['mut_prob'] = max(0.05, min(params['mut_prob'], 0.3))
        params['tournament_size'] = max(3, min(params['tournament_size'], 10))
    
    # 最終結果報告
    print("\n\n===============================================")
    print("          Auto-tuning Results Summary")
    print("===============================================")
    for i, res in enumerate(trial_results):
        status = "✅" if res['cost'] <= target_cost else "❌"
        print(f"Trial {i+1}: Cost = {res['cost']:.2f}, Time = {res['time']:.2f}s {status}")
        print(f"    Params: Pop={res['params']['population_size']}, Gen={res['params']['max_gens']}, "
              f"Cx={res['params']['cx_prob']:.2f}, Mut={res['params']['mut_prob']:.2f}, "
              f"Tourn={res['params']['tournament_size']}")
    
    if best_cost <= target_cost:
        print(f"\n🎉 Successfully achieved target cost of {target_cost}!")
    else:
        print(f"\n⚠️ Failed to achieve target cost. Best cost: {best_cost:.2f}")
    
    # 使用最佳結果進行最終繪圖
    if best_result and tsp_instance.plot_enabled:
        print("\nGenerating final visualization with best solution...")
        anim = AnimationTSP(
            best_result['solutions_history'], 
            tsp_instance.coord_x, 
            tsp_instance.coord_y, 
            best_result['costs_history']
        )
        anim.animate_routes()
        
        # 繪製適應度進化圖
        plot_evolution(best_result['min_values'], best_result['avg_values'])
    
    return best_result

def plot_evolution(min_values, avg_values):
    """
    繪製適應度進化圖
    """
    plt.figure(figsize=(10, 6))
    plt.plot(min_values, 'g-', label='Best Cost')
    plt.plot(avg_values, 'b--', label='Average Cost')
    plt.title('Evolution of Population Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    主函數，提供用戶交互菜單
    """
    print("=========================================")
    print("  TSP GA Solver - Auto-tuning Mode")
    print("=========================================")
    
    # 獲取用戶輸入
    instance_file = input("Enter TSP instance file path (e.g., ch150.tsp): ").strip()
    try:
        target_cost = float(input("Enter target cost: "))
    except ValueError:
        print("Invalid target cost. Using default 10000.")
        target_cost = 10000
    
    max_trials = 10
    try:
        max_trials = int(input(f"Enter max trials [default={max_trials}]: ") or max_trials)
    except ValueError:
        pass
    
    global use_numpy
    use_numpy = int(input("Use numpy arrays? (0=No, 1=Yes) [default=0]: ") or 0)
    
    # 運行自動調整
    auto_tune_ga(instance_file, target_cost, max_trials)

if __name__ == "__main__":
    main()
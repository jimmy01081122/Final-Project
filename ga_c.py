# ===============================================================
#  File Name        : ga.py
#  Description      : TSP Problem Solver using Genetic Algorithm (GA)
#  Origin Source    : https://github.com/RenatoMaynard/TSP-Genetic-Algorithm/blob/main/ga_interactive.py
#  Original Author  : Renato Maynard
#  Modified by      : Jimmy Chang
#  Source           : DEAP, matplotlib, tsplib95 , numpy, IPython
#  Version          : 1.0.0
#  Update Date      : 2025-06-12
#  Python Version   : 3.8+
#  License          : MIT
#  Run Command      : python3 ga_c.py
# ===============================================================
# ===================================================================
#                      導入必要的函式庫
# ===================================================================
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import numpy as np
import tsplib95
import random
import time
from deap import algorithms, base, creator, tools
import json
import logging
import sys
import builtins
import os

# 提高在 Jupyter Notebook 中嵌入動畫的容量限制，避免動畫過大無法顯示
plt.rcParams['animation.embed_limit'] = 2**128

# ===================================================================
#                         全域變數 (Global Variables)
# ===================================================================
INF = 9999999  # 代表無窮大，用於初始化距離矩陣
dist_matrix = []  # 全域距離矩陣，儲存城市間的距離
use_numpy = 0  # 控制個體(染色體)的資料結構: 0=使用Python原生list, 1=使用Numpy陣列

# ===================================================================
#          日誌設定 (Logging): 自動將 print 內容同步存檔
# ===================================================================
log_filename = "tsp_solver.log"  # Log 檔案名稱
logging.basicConfig(
    level=logging.INFO,  # 設定日誌級別為 INFO
    format="%(asctime)s  %(levelname)s  %(message)s",  # 設定日誌格式
    handlers=[
        # 設定日誌檔案處理器，模式為 'a' (append)
        logging.FileHandler(log_filename, mode="a", encoding="utf-8"),
        # 同時也將日誌輸出到螢幕
        logging.StreamHandler(sys.stdout)
    ]
)

# 備份原生的 print 函式
_builtin_print = builtins.print
def print(*args, **kwargs):
    """
    客製化的 print 函式。
    作用：除了在螢幕上印出訊息外，同時也將訊息寫入日誌檔案。
    """
    _builtin_print(*args, **kwargs)  # 執行原生 print 功能，在終端顯示
    logging.info(" ".join(map(str, args)))  # 將訊息寫入 log 檔案

# 用我們客製化的版本取代全域的 print 函式
builtins.print = print
# ===================================================================


class ParameterTuner:
    """
    類別：參數調諧器 (ParameterTuner)
    作用：處理遺傳演算法 (GA) 的參數自動調校。根據每次執行的成本，動態調整下一輪的參數。
    """
    def __init__(self, target_cost, max_attempts=50, improvement_threshold=0.95):
        """
        初始化函式
        輸入:
            - target_cost (float): 希望達成的目標成本。
            - max_attempts (int): 最大嘗試調校次數。
            - improvement_threshold (float): 改善率閾值，若改善太少則可能採取更激進的調參策略。
        """
        self.target_cost = target_cost
        self.max_attempts = max_attempts
        self.improvement_threshold = improvement_threshold

        # 定義各參數可供選擇的範圍
        self.param_ranges = {
            'population_size': [50, 100, 200, 300, 500],       # 族群大小
            'max_gens': [500, 1000, 1500, 2000, 3000],          # 最大世代數
            'cx_prob': [0.7, 0.8, 0.9, 0.95],                   # 交配率 (Crossover Probability)
            'mut_prob': [0.05, 0.1, 0.15, 0.2, 0.3],            # 突變率 (Mutation Probability)
            'tournament_size': [2, 3, 4, 5, 6, 8]               # 錦標賽選擇的規模
        }

        # 設定一組初始參數
        self.current_params = {
            'population_size': 100,
            'max_gens': 1000,
            'cx_prob': 0.9,
            'mut_prob': 0.1,
            'tournament_size': 4
        }

        self.best_cost = float('inf')  # 記錄歷史上找到的絕對最低成本
        self.best_params = self.current_params.copy()  # 記錄產生最低成本的參數組合
        self.attempt_history = []  # 記錄每一次嘗試的參數和結果

    def adjust_parameters(self, current_cost, attempt_num):
        """
        函式：調整參數 (adjust_parameters)
        作用：根據當前的執行結果（成本）來調整下一輪的GA參數。
        輸入:
            - current_cost (float): 本次GA執行的最終成本。
            - attempt_num (int): 當前的嘗試次數。
        輸出:
            - (bool): 如果成本**正好等於**目標，返回 True，表示調校結束；否則返回 False。
        """
        print(f"\nAttempt {attempt_num}: Current Cost = {current_cost}, Target = {self.target_cost}")

        # 將這次的嘗試結果記錄下來
        self.attempt_history.append({
            'attempt': attempt_num,
            'params': self.current_params.copy(),
            'cost': current_cost
        })

        # 如果本次成本比歷史最佳成本還低，就更新最佳紀錄
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_params = self.current_params.copy()
            print(f"New best cost found: {self.best_cost}")

        # 【本次修改的核心】
        # 只有在成本「完全等於」目標成本時，才返回 True
        if current_cost == self.target_cost:
             print(f"🎉 Exact target reached! Found a solution with cost {self.target_cost}.")
             return True

        # 計算改善率，判斷是否陷入局部最優
        improvement_ratio = self.best_cost / current_cost if current_cost > 0 else 0

        # 如果改善不明顯且已經嘗試多次，則採取較激進的參數調整策略
        if improvement_ratio < self.improvement_threshold and attempt_num > 5:
            print("Improvement is low. Applying aggressive parameter change.")
            self._aggressive_parameter_change()
        else:
            # 否則，進行常規的參數微調
            self._normal_parameter_adjustment(current_cost)

        print(f"Adjusted parameters: {self.current_params}")
        return False # 未找到精確目標，繼續嘗試

    def _normal_parameter_adjustment(self, current_cost):
        """
        函式：常規參數調整
        作用：根據目前成本與目標的差距，進行穩健的參數調整。
        """
        # 計算當前成本與目標成本的比率
        cost_ratio = current_cost / self.target_cost if self.target_cost > 0 else 1

        # 如果成本遠高於目標 (超過1.5倍)，增加探索能力
        if cost_ratio > 1.5:
            # 增加族群大小和世代數，以進行更廣泛的搜索
            try:
                current_pop_idx = self.param_ranges['population_size'].index(self.current_params['population_size'])
                if current_pop_idx < len(self.param_ranges['population_size']) - 1:
                    self.current_params['population_size'] = self.param_ranges['population_size'][current_pop_idx + 1]
            except ValueError: pass # 如果當前值不在列表中，則跳過

            try:
                current_gen_idx = self.param_ranges['max_gens'].index(self.current_params['max_gens'])
                if current_gen_idx < len(self.param_ranges['max_gens']) - 1:
                    self.current_params['max_gens'] = self.param_ranges['max_gens'][current_gen_idx + 1]
            except ValueError: pass

            # 提高突變率，增加多樣性
            try:
                current_mut_idx = self.param_ranges['mut_prob'].index(self.current_params['mut_prob'])
                if current_mut_idx < len(self.param_ranges['mut_prob']) - 1:
                    self.current_params['mut_prob'] = self.param_ranges['mut_prob'][current_mut_idx + 1]
            except ValueError: pass

        # 如果成本接近目標 (1.1倍以上)，進行微調以求精進
        elif cost_ratio > 1.1:
            # 微調交配率和選擇壓力（錦標賽規模）
            try:
                current_cx_idx = self.param_ranges['cx_prob'].index(self.current_params['cx_prob'])
                if current_cx_idx < len(self.param_ranges['cx_prob']) - 1:
                    self.current_params['cx_prob'] = self.param_ranges['cx_prob'][current_cx_idx + 1]
            except ValueError: pass

            try:
                current_tourn_idx = self.param_ranges['tournament_size'].index(self.current_params['tournament_size'])
                if current_tourn_idx < len(self.param_ranges['tournament_size']) - 1:
                    self.current_params['tournament_size'] = self.param_ranges['tournament_size'][current_tourn_idx + 1]
            except ValueError: pass

    def _aggressive_parameter_change(self):
        """
        函式：激進參數調整
        作用：當演算法可能陷入停滯時，隨機地從預設範圍中選擇新參數，以跳出局部最優。
        """
        # 對每個參數，有 60% 的機率從其範圍內隨機挑選一個新值
        for param, ranges in self.param_ranges.items():
            if random.random() < 0.6:
                self.current_params[param] = random.choice(ranges)

        # 確保族群和世代數不會太小，以應對困難問題
        if self.current_params['population_size'] < 200:
             self.current_params['population_size'] = random.choice([p for p in self.param_ranges['population_size'] if p >= 200])
        if self.current_params['max_gens'] < 1500:
            self.current_params['max_gens'] = random.choice([g for g in self.param_ranges['max_gens'] if g >= 1500])

    def save_tuning_history(self, filename="tuning_history.json"):
        """
        函式：儲存調校歷史
        作用：將整個調校過程中的所有嘗試（參數、成本）以及最終找到的最佳結果保存到 JSON 檔案中。
        """
        history_data = {
            'best_params': self.best_params,
            'best_cost': self.best_cost,
            'target_cost': self.target_cost,
            'attempt_history': self.attempt_history
        }
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=4) # indent=4 讓 JSON 檔案格式更美觀
        print(f"Tuning history saved to {filename}")


class AnimationTSP:
    """
    類別：TSP 動畫產生器
    作用：將GA在求解過程中路線的演進過程視覺化成動畫。
    """
    def __init__(self, history, x_coords, y_coords, costs):
        """
        初始化函式
        輸入:
            - history (list): 一個包含多個解的列表，每個解都是一個城市索引的順序列表。
            - x_coords (list): 所有城市的 X 座標。
            - y_coords (list): 所有城市的 Y 座標。
            - costs (list): 對應 history 中每個解的成本。
        """
        self.history = [list(h) for h in history]
        self.costs = costs
        self.points = np.column_stack((x_coords, y_coords))
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.line, = self.ax.plot([], [], lw=2, color='blue')
        self.title = self.ax.text(
            0.5, 1.05, "",
            bbox={'facecolor': 'w', 'alpha': 0.8, 'pad': 5},
            transform=self.ax.transAxes,
            ha="center", fontsize=12
        )

    def init_animation(self):
        """初始化動畫背景，只在第一幀執行。"""
        self.ax.plot(self.points[:, 0], self.points[:, 1], 'ro', markersize=4)
        margin_x = (self.points[:, 0].max() - self.points[:, 0].min()) * 0.1
        margin_y = (self.points[:, 1].max() - self.points[:, 1].min()) * 0.1
        self.ax.set_xlim(self.points[:, 0].min() - margin_x, self.points[:, 0].max() + margin_x)
        self.ax.set_ylim(self.points[:, 1].min() - margin_y, self.points[:, 1].max() + margin_y)
        self.line.set_data([], [])
        return self.line,

    def update_animation(self, frame_idx):
        """為動畫的每一幀更新繪圖內容。"""
        route = self.history[frame_idx]
        cost = self.costs[frame_idx]
        ordered_points = self.points[route + [route[0]]]
        self.line.set_data(ordered_points[:, 0], ordered_points[:, 1])
        self.title.set_text(f"Generation: {frame_idx}, Cost: {cost:.2f}")
        return self.line, self.title

    def animate_routes(self):
        """創建並顯示動畫。"""
        num_frames = min(len(self.history), 200)
        frame_indices = np.linspace(0, len(self.history) - 1, num_frames, dtype=int)
        self.history = [self.history[i] for i in frame_indices]
        self.costs = [self.costs[i] for i in frame_indices]

        ani = FuncAnimation(
            self.fig, self.update_animation, frames=len(self.history),
            init_func=self.init_animation, blit=False, interval=100, repeat=False
        )
        plt.close(self.fig)
        display(HTML(ani.to_jshtml()))


class TSPInstance:
    """
    類別：TSP 問題實例
    作用：從 tsplib95 格式的檔案中載入 TSP 問題，並準備好座標和距離矩陣等數據。
    """
    def __init__(self, plot_route, instance_file):
        """
        初始化函式
        輸入:
            - plot_route (bool): 是否要載入座標以供後續繪圖。
            - instance_file (str): TSPLIB 格式檔案的路徑。
        """
        self.plot_enabled = bool(plot_route)
        self.coord_x = []
        self.coord_y = []

        try:
            self.problem = tsplib95.load(instance_file)
            self.info = self.problem.as_keyword_dict()
            self.n = len(list(self.problem.get_nodes()))
        except Exception as e:
            print(f"Error loading TSP file: {e}")
            sys.exit(1)

        if self.plot_enabled and self._can_plot():
            node_coords = self.info.get('NODE_COORD_SECTION', {})
            for i in range(1, self.n + 1):
                if i in node_coords:
                    x, y = node_coords[i]
                    self.coord_x.append(x)
                    self.coord_y.append(y)
        else:
            self.plot_enabled = False

    def _can_plot(self):
        """檢查 TSP 問題的權重類型是否支援二維座標繪圖。"""
        dist_type = self.info.get('EDGE_WEIGHT_TYPE')
        if dist_type in ['EUC_2D', 'GEO', 'ATT']:
            return True
        else:
            print(f"Warning: Plotting is not supported for this EDGE_WEIGHT_TYPE ({dist_type}).")
            return False

    def generate_distance_matrix(self):
        """產生全域的城市距離矩陣 (dist_matrix)。"""
        global dist_matrix
        dist_matrix = [[INF] * self.n for _ in range(self.n)]
        nodes = list(self.problem.get_nodes())
        for i_idx, node_i in enumerate(nodes):
            for j_idx, node_j in enumerate(nodes):
                if i_idx != j_idx:
                    dist_matrix[i_idx][j_idx] = self.problem.get_weight(node_i, node_j)


def distance(i, j):
    """函式：查詢距離。從全域距離矩陣中返回城市 i 和 j 之間的距離。"""
    return dist_matrix[i][j]


def total_cost(route):
    """
    函式：計算總成本 (適應度函式)
    作用：計算一條給定 TSP 路徑的總長度。
    """
    cost = sum(distance(route[i], route[i+1]) for i in range(len(route)-1))
    cost += distance(route[-1], route[0])
    return (cost,)


def nearest_neighbor(n):
    """
    函式：最近鄰居法 (用於產生初始個體)
    作用：以一定機率使用「最近鄰居」啟發式演算法生成一條初始路徑，否則隨機生成。
    """
    if random.random() < 0.4:
        start_node = random.randrange(n)
        unvisited = set(range(n))
        unvisited.remove(start_node)
        route = [start_node]
        current_node = start_node
        while unvisited:
            next_node = min(unvisited, key=lambda city: distance(current_node, city))
            unvisited.remove(next_node)
            route.append(next_node)
            current_node = next_node
    else:
        route = list(range(n))
        random.shuffle(route)
    return np.array(route) if use_numpy else route


def mutate(individual):
    """
    函式：突變操作 (Mutation)
    作用：對一個個體（路徑）進行突變，此處使用「反轉子路徑」策略。
    """
    size = len(individual)
    i, j = random.sample(range(size), 2)
    if i > j:
        i, j = j, i
    sub_route = individual[i:j+1]
    sub_route.reverse()
    individual[i:j+1] = sub_route
    return individual,


def setup_deap_toolbox(n_cities):
    """設置 DEAP 工具箱的輔助函式"""
    if "FitnessMin" in creator.__dict__: del creator.FitnessMin
    if "Individual" in creator.__dict__: del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    IndividualType = np.ndarray if use_numpy else list
    creator.create("Individual", IndividualType, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", nearest_neighbor, n_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", total_cost)
    toolbox.register("select", tools.selTournament)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", mutate)
    return toolbox


def ga_simple(tsp_instance, seed):
    """
    簡易版 GA
    """
    population_size = 50
    max_gens = 200
    cx_prob = 0.9
    mut_prob = 0.4
    tournament_size = 4
    
    random.seed(seed)
    np.random.seed(seed)

    toolbox = setup_deap_toolbox(tsp_instance.n)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    start_time = time.time()
    final_population, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=max_gens,
        stats=stats, halloffame=hof, verbose=True
    )
    end_time = time.time()

    min_list, avg_list = logbook.select("min", "avg")
    print(f"\nBest route cost: {min(min_list):.2f}")
    print(f"Execution time: {end_time - start_time:.2f}s")

    if tsp_instance.plot_enabled:
        plot_evolution(min_list, avg_list)
        final_route = list(hof[0])
        plot_static_route(final_route, tsp_instance.coord_x, tsp_instance.coord_y, min(min_list), " (Simple GA Best)")


def ga_advanced(tsp_instance, seed):
    """
    進階版 GA (手動演化迴圈)
    """
    population_size = 100
    max_gens = 1000
    cx_prob = 0.9
    mut_prob = 0.1
    tournament_size = 4

    random.seed(seed)
    np.random.seed(seed)

    toolbox = setup_deap_toolbox(tsp_instance.n)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "avg", "std", "max"

    start_time = time.time()
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)

    solutions_history = [list(hof[0])]
    costs_history = [hof[0].fitness.values[0]]

    for gen in range(1, max_gens + 1):
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

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)
        
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        solutions_history.append(list(hof[0]))
        costs_history.append(hof[0].fitness.values[0])

    end_time = time.time()
    best_cost = hof[0].fitness.values[0]
    
    print(f"\nBest route cost: {best_cost:.2f}")
    print(f"Execution time: {end_time - start_time:.2f}s")

    if tsp_instance.plot_enabled:
        plot_evolution(logbook.select("min"), logbook.select("avg"))
        final_route = list(hof[0])
        plot_static_route(final_route, tsp_instance.coord_x, tsp_instance.coord_y, best_cost, " (Advanced GA Best)")
        # AnimationTSP(solutions_history, tsp_instance.coord_x, tsp_instance.coord_y, costs_history).animate_routes()


def plot_evolution(min_values, avg_values):
    """
    繪製成本演化圖
    """
    plt.figure()
    plt.plot(min_values, 'c-', label='Best Cost')
    plt.plot(avg_values, 'b-', label='Average Cost')
    plt.legend(frameon=True)
    plt.ylabel('Cost')
    plt.xlabel('Generations')
    plt.title("Cost Evolution over Generations")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def ga_tunable(tsp_instance, params, seed, verbose=True):
    """
    函式：可調參數的遺傳演算法主體 (模式1使用)
    """
    population_size = int(params['population_size'])
    max_gens = int(params['max_gens'])
    cx_prob = params['cx_prob']
    mut_prob = params['mut_prob']
    tournament_size = int(params['tournament_size'])
    
    toolbox = setup_deap_toolbox(tsp_instance.n)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=max_gens,
        stats=stats, halloffame=hof, verbose=verbose
    )

    costs_history = logbook.select("min")
    final_best_solution = hof[0]
    solutions_history = [final_best_solution] * len(costs_history)
    best_cost = hof[0].fitness.values[0]

    return best_cost, solutions_history, costs_history


def self_tuning_tsp_solver(instance_file, target_cost, max_attempts=50):
    """
    函式：自調諧 TSP 求解器 (模式1的主體)
    """
    print("="*60)
    print(" 🤖 Mode 1: Find Specific Cost (Self-Tuning Solver)")
    print("="*60)
    
    global use_numpy
    use_numpy = 0

    tsp_instance = TSPInstance(plot_route=False, instance_file=instance_file)
    tsp_instance.generate_distance_matrix()
    print(f"Loaded TSP instance with {tsp_instance.n} cities.")

    tuner = ParameterTuner(target_cost, max_attempts)

    best_overall_cost = float('inf')
    best_overall_params = None
    best_route_history = None
    
    best_diff = float('inf')
    closest_cost = float('inf')
    closest_params = None
    closest_route_hist = None

    target_achieved = False

    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*25} Attempt {attempt}/{max_attempts} {'='*25}")
        seed = int(time.time()) + attempt

        try:
            current_cost, sol_hist, cost_hist = ga_tunable(
                tsp_instance, tuner.current_params, seed, verbose=False
            )

            if current_cost < best_overall_cost:
                best_overall_cost = current_cost
                best_overall_params = tuner.current_params.copy()
                best_route_history = sol_hist

            diff = abs(current_cost - target_cost)
            if diff < best_diff:
                best_diff = diff
                closest_cost = current_cost
                closest_params = tuner.current_params.copy()
                closest_route_hist = sol_hist

            if tuner.adjust_parameters(current_cost, attempt):
                target_achieved = True
                best_overall_cost = current_cost
                best_overall_params = tuner.current_params.copy()
                best_route_history = sol_hist
                break

        except Exception as e:
            print(f"An error occurred in attempt {attempt}: {e}")
            tuner._aggressive_parameter_change()
            continue
            
    print("\n" + "="*60)
    print(" 🎯 Self-Tuning Final Report")
    print("="*60)

    plot_instance = TSPInstance(plot_route=True, instance_file=instance_file)
    if not plot_instance.plot_enabled:
        print("Cannot plot results as this instance type is not plottable.")
    else:
        xs, ys = plot_instance.coord_x, plot_instance.coord_y
        if target_achieved:
            print(f"✅ Exact target reached! Displaying result for cost: {best_overall_cost:.2f}")
            final_route = list(best_route_history[-1])
            plot_static_route(final_route, xs, ys, best_overall_cost, " (Exact Target Met)")
        else:
            print(f"⚠️ Exact target not met. Displaying closest result found: {closest_cost:.2f} (Difference: {best_diff:.2f})")
            final_route = list(closest_route_hist[-1])
            plot_static_route(final_route, xs, ys, closest_cost, " (Closest to Target)")

    print("-" * 60)
    if target_achieved:
        print(f"✅ Exact Target Reached!")
        print("-" * 60)
        print(f"  - Achieved Cost: {best_overall_cost:.2f}")
        print(f"  - Parameters Used: {best_overall_params}")
    else:
        print(f"⚠️ Exact Target Not Met.")
        print("-" * 60)
        print("--- [Closest Result to Target] ---")
        print(f"  - Cost: {closest_cost:.2f} (Difference from target: {best_diff:.2f})")
        print(f"  - Parameters Used: {closest_params}")
        print("\n--- [Best Overall Result (Lowest Cost)] ---")
        print(f"  - Cost: {best_overall_cost:.2f}")
        print(f"  - Parameters Used: {best_overall_params}")

    print("-" * 60)
    print(f"  - Target Cost: {target_cost:.2f}")
    print(f"  - Total Attempts: {len(tuner.attempt_history)}")
    print("="*60)

    tuner.save_tuning_history()

    print("\nGenerating Parameter vs. Cost plots...")
    if tuner.attempt_history:
        param_names = tuner.attempt_history[0]['params'].keys()
        for p_name in param_names:
            try:
                x_vals = [rec['params'][p_name] for rec in tuner.attempt_history]
                y_vals = [rec['cost'] for rec in tuner.attempt_history]
                plt.figure(figsize=(8, 5))
                plt.scatter(x_vals, y_vals, alpha=0.7, edgecolors='k')
                plt.xlabel(p_name.replace('_', ' ').title())
                plt.ylabel('Resulting Cost')
                plt.title(f'Cost vs. {p_name.replace("_", " ").title()}')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not plot for parameter '{p_name}': {e}")


def plot_static_route(route, x_coords, y_coords, cost, title_suffix=""):
    """
    函式：繪製靜態 TSP 路線圖
    """
    route_with_return = route + [route[0]]
    xs = [x_coords[i] for i in route_with_return]
    ys = [y_coords[i] for i in route_with_return]
    plt.figure(figsize=(10, 10))
    plt.plot(xs, ys, 'b-', lw=1.5, label='Route')
    plt.plot(x_coords, y_coords, 'ro', markersize=5, label='Cities')
    plt.plot(xs[0], ys[0], 'go', markersize=10, label='Start/End')
    plt.title(f"TSP Route | Cost = {cost:.2f}{title_suffix}", fontsize=16)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def run_self_tuning_mode():
    """模式1的執行函式"""
    try:
        instance_file = input("Enter the file path of the TSP instance (e.g., kroA100.tsp): ").strip()
        target_cost = float(input("Enter target cost to achieve: "))
        max_attempts_str = input("Enter maximum tuning attempts (default: 30): ").strip()
        max_attempts = int(max_attempts_str) if max_attempts_str else 30
        self_tuning_tsp_solver(instance_file, target_cost, max_attempts)
    except FileNotFoundError:
        print(f"\nError: File '{instance_file}' not found. Please check the path.")
    except ValueError:
        print("\nError: Invalid input. Please enter a valid number for cost and attempts.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

def run_find_best_mode():
    """模式2的執行函式"""
    global use_numpy
    try:
        instance_file = input("Enter the file path of the TSP instance (e.g., kroA100.tsp): ").strip()
        seed_value = int(input("Enter random seed (integer), e.g. 42: "))
        version = input("Choose GA version (0=Simple, 1=Advanced): ").strip()
        use_numpy = int(input("Use numpy arrays for individuals? (0=No, 1=Yes): "))

        tsp_instance = TSPInstance(plot_route=True, instance_file=instance_file)
        tsp_instance.generate_distance_matrix()

        if version == '0':
            print("\nRunning Simple GA...")
            ga_simple(tsp_instance, seed_value)
        elif version == '1':
            print("\nRunning Advanced GA...")
            ga_advanced(tsp_instance, seed_value)
        else:
            print("Invalid GA version selected.")
    except FileNotFoundError:
        print(f"\nError: File '{instance_file}' not found. Please check the path.")
    except ValueError:
        print("\nError: Invalid input. Please enter valid integers.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


def main():
    """
    主執行函式
    """
    print("="*40)
    print("      TSP GA Solver - Interactive Menu")
    print("="*40)
    mode = input("choose mode : (find cost = 1) (find best = 2) ")

    if mode == '1':
        run_self_tuning_mode()
    elif mode == '2':
        run_find_best_mode()
    else:
        print("Invalid mode selected. Please restart and choose 1 or 2.")


if __name__ == "__main__":
    """
    Python 的主程式入口。
    """
    main()

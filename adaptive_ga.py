import random
import tsp


def adaptive_ga(instance_file, target_cost, seed=None, max_attempts=5):
    """Run the GA repeatedly adjusting parameters until target_cost is reached.

    Parameters
    ----------
    instance_file : str
        Path to the TSPLIB instance file.
    target_cost : float
        Desired tour cost to reach.
    seed : int, optional
        Initial random seed. If ``None`` a random seed is used.
    max_attempts : int, optional
        Maximum number of parameter adjustment cycles.
    """
    if seed is None:
        seed = random.randint(0, 1000000)

    tsp_instance = tsp.TSPInstance(plot_route=0, instance_file=instance_file)
    tsp_instance.generate_distance_matrix()

    params = {
        "population_size": 100,
        "max_gens": 1000,
        "cx_prob": 0.9,
        "mut_prob": 0.1,
    }

    best_cost = float("inf")
    attempt = 0
    while best_cost > target_cost and attempt < max_attempts:
        print(f"Attempt {attempt + 1} with parameters {params}")
        best_cost = tsp.ga_advanced_params(
            tsp_instance,
            seed,
            population_size=params["population_size"],
            max_gens=params["max_gens"],
            cx_prob=params["cx_prob"],
            mut_prob=params["mut_prob"],
        )
        if best_cost <= target_cost:
            print(f"Target cost reached: {best_cost}")
            break

        attempt += 1
        params["population_size"] = int(params["population_size"] * 1.5)
        params["max_gens"] += 500
        params["mut_prob"] = min(0.5, params["mut_prob"] + 0.05)

    if best_cost > target_cost:
        print(f"Target not reached. Best cost: {best_cost}")
    return best_cost


if __name__ == "__main__":
    instance_path = input("Enter the TSP instance path: ")
    target = float(input("Enter desired tour cost: "))
    seed_input = input("Random seed (leave blank for random): ")
    if seed_input.strip():
        seed_val = int(seed_input)
    else:
        seed_val = None
    adaptive_ga(instance_path, target, seed_val)

import numpy as np
from tqdm import tqdm
import pandas as pd

def load_data(filename):
    with open(filename, "r") as f:
        V = []
        data = f.readlines()
        for i, line in enumerate(data):
            if line.startswith("NODE_COORD_SECTION"):
                data = data[i+1:len(data)-1]
                break
        for i, line in enumerate(data):
            node = line.split()[1:]
            node = np.array([float(i) for i in node])
            V.append(node)
        cities = {i : V[i] for i in range(len(V))}
        return cities

def euclid_distance(x,y):
    # x and y are numpy arrays
    return np.sqrt(np.sum((x-y)**2))

def calculate_paths(cities:dict):
    names = [i for i in range(len(cities))]
    paths = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                paths[i,j] = euclid_distance(cities[names[i]], cities[names[j]])
    return paths

def initialize_tour(paths):
    """
    We choose the tour using the nearest neighbor algorithm
    """

    num_cities = paths.shape[0]
    visited = [False] * num_cities
    tour = [np.random.randint(0, num_cities)]
    visited[tour[0]] = True

    while len(tour) < num_cities:
        current_city = tour[-1]
        nearest_city = None
        nearest_distance = float('inf')

        # Find the nearest unvisited city to the current city
        for city in range(num_cities):
            if not visited[city] and paths[current_city, city] < nearest_distance:
                nearest_city = city
                nearest_distance = paths[current_city, city]

        tour.append(nearest_city)
        visited[nearest_city] = True

    return tour

def generate_neighbors(x, n=20):
    neighbors = []
    while len(neighbors) < n:
        i = np.random.randint(0, len(x)-2)
        j = np.random.randint(1, len(x)-1)
        if i > j :
            i, j = j, i
        neighbor = x.copy()
        neighbor[i:j] = neighbor[i:j][::-1]
        if tuple(neighbor) not in map(tuple, neighbors):
            neighbors.append(neighbor)
    return neighbors

def fitness(x, paths):
    # x is a list of cities
    # paths is a matrix of distances between cities
    fitness = 0
    for i in range(len(x)-1):
        fitness += paths[x[i], x[i+1]]
    fitness += paths[x[-1], x[0]]
    return fitness

def best_neighbor(x:list, paths:np.array, generate_neighbors:callable = generate_neighbors, fitness: callable = fitness):
    neighbors = generate_neighbors(x)
    best_neighbor = neighbors[0]
    for neighbor in range(1, len(neighbors)):
        if fitness(neighbors[neighbor], paths) < fitness(best_neighbor, paths):
            best_neighbor = neighbors[neighbor]
    return best_neighbor

def random_neighbor(x:list, paths:np.array, generate_neighbors:callable = generate_neighbors):
    neighbors = generate_neighbors(x)
    return neighbors[np.random.randint(0, len(neighbors))]

def hill_climbing(f:callable, x_init:float, n_iters:int, paths:np.array, variant:str, epsilon:float = 0.001, steepest:bool = False):
    # choose intial x randomly as x_best
    x = x_init
    x_best = x
    if variant == "simple":
        neighbor_function = best_neighbor
    elif variant == "stochastic":
        neighbor_function = random_neighbor
    for iter in tqdm(range(n_iters)):
        y = neighbor_function(x, paths)
        if f(x, paths) > f(y, paths) :
            x = y
            if f(x, paths) < f(x_best, paths):
                x_best = x
            else:
                if steepest:
                    x = x_best
    return x_best

if __name__ == "__main__":
    filenames = ["data/rd100.tsp", "data/eil101.tsp",
                 "data/a280.tsp", "data/d198.tsp", "data/ch150.tsp"]
    results = {}
    for name in filenames:
        cities = load_data(name)
        paths = calculate_paths(cities)
        x = initialize_tour(paths)
        print(fitness(x, paths))
        sol_ste = hill_climbing(fitness, x, 10000, paths, "simple", steepest=True)
        sol_sto = hill_climbing(fitness, x, 10000, paths, "stochastic")
        sol_sim = hill_climbing(fitness, x, 10000, paths, "simple")
        fit_ste = fitness(sol_ste, paths)
        fit_sto = fitness(sol_sto, paths)
        fit_sim = fitness(sol_sim, paths)
        results[name[9:]] = [fit_ste, fit_sto, fit_sim]
    pd.DataFrame(results).to_csv("Desktop/results.csv")

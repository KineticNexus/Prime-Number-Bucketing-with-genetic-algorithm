import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime
from deap import base, creator, tools, algorithms
import random

def help():
    """Spiral Generation Parameters:
    
    'A_MIN' (18) and 'A_MAX' (25):
    These define the range for the 'a' parameter in spiral generation.
    'a' is the starting radius of the spiral.
    A higher 'a' value will start the spiral further from the center.
    This range (18 to 25) will result in spirals that start relatively far from the center.
    Effect: Controls the initial size of the spiral.
    
    'B_MIN' (0.0001) and 'B_MAX' (0.01):
    These define the range for the 'b' parameter in spiral generation.
    'b' determines how quickly the spiral expands as it rotates.
    This range is quite small, meaning the spirals will expand very slowly.
    Effect: Controls how tightly wound the spiral is. With these values, you'll get very tight spirals.
    
    'C_MIN' (0.001) and 'C_MAX' (200):
    These define the range for the 'c' parameter in spiral generation.
    'c' determines the number of rotations in the spiral.
    This wide range allows for spirals with very few to many rotations.
    Effect: Controls how many times the spiral goes around the center.
    
    'POINTS_MIN' (108) and 'POINTS_MAX' (15000):
    These define the range for the number of points in the spiral.
    This is a wide range, allowing for both sparse and very dense spirals.
    Effect: Controls the resolution of the spiral and how many prime numbers are considered.
    
    Genetic Algorithm Parameters:
    
    'POPULATION_SIZE' (300):
    The number of individuals in each generation.
    A larger population allows for more diversity but increases computation time.
    Effect: Increases the chance of finding good solutions but slows down the algorithm.
    
    'P_CROSSOVER' (0.5):
    The probability of crossover between two individuals.
    At 0.5, about half of the offspring will be created through crossover.
    Effect: Controls the rate of mixing between solutions.
    
    'P_MUTATION' (0.35):
    The probability of mutation for each individual.
    This is a relatively high mutation rate, promoting exploration of the solution space.
    Effect: Increases diversity and helps avoid local optima.
    
    'MAX_GENERATIONS' (3000):
    The maximum number of generations the algorithm will run.
    3000 is a high number, allowing for extensive optimization.
    Effect: Determines how long the algorithm will run and optimize.
    
    'TOURNAMENT_SIZE' (3):
    The number of individuals that compete in each tournament selection.
    A size of 3 provides a balance between selection pressure and diversity maintenance.
    Effect: Influences how strongly the algorithm favors better solutions.
    
    'ETA' (5):
    The crowding degree of crossover and mutation.
    A lower value (5) will produce offspring less similar to their parents.
    Effect: Controls how different offspring are from their parents.
    
    'INDPB' (0.05):
    The probability of each attribute to be mutated.
    At 0.05, there's a 5% chance for each parameter to be mutated independently.
    Effect: Fine-tunes the exploration of the solution space.
    
    Visualization and Fitness Parameters:
    
    'PLOT_REFRESH' (5):
    The number of generations between plot updates.
    A lower value (5) means more frequent updates, which can slow down the algorithm but provides more real-time feedback.
    Effect: Controls how often the visualization is updated.
    
    'POINTS_WEIGHT' (0.2):
    The weight given to the number of points in the fitness function.
    At 0.2, it gives some importance to having more points, but still prioritizes empty buckets.
    Effect: Balances the trade-off between maximizing empty buckets and using more points.
    
    Adaptive Mutation Parameters:
    
    'STAGNATION_THRESHOLD' (50):
    Number of generations without improvement before increasing mutation rate.
    Effect: Determines how quickly the algorithm responds to lack of progress.
    
    'MUTATION_INCREASE' (0.05):
    How much to increase the mutation rate when stagnation is detected.
    Effect: Controls the aggressiveness of the adaptive mutation strategy.
    
    'MAX_MUTATION_RATE' (0.2):
    Maximum allowed mutation rate.
    Effect: Prevents the mutation rate from becoming too high, which could make the search too random."""
    
# Parameters
PARAMS = {
    # Spiral generation parameters
    'A_MIN': 45,      # Minimum value for parameter 'a' in spiral generation
    'A_MAX': 55,       # Maximum value for parameter 'a' in spiral generation
    'B_MIN': 0.0001,     # Minimum value for parameter 'b' in spiral generation
    'B_MAX': 0.001,        # Maximum value for parameter 'b' in spiral generation
    'C_MIN': 60,    # Minimum value for parameter 'c' in spiral generation
    'C_MAX': 90,      # Maximum value for parameter 'c' in spiral generation
    'POINTS_MIN': 108,  # Minimum number of points in the spiral
    'POINTS_MAX': 5000,  # Maximum number of points in the spiral
    # Genetic Algorithm parameters
    'POPULATION_SIZE': 1300,  # Number of individuals in the population
    'P_CROSSOVER': 0.5,      # Probability of crossover
    'P_MUTATION': 0.05,       # Probability of mutation
    'MAX_GENERATIONS': 3000,  # Maximum number of generations
    'TOURNAMENT_SIZE': 10,    # Size of tournament for selection
    'ETA': 5,               # Crowding degree of crossover and mutation
    'INDPB': 0.3,            # Probability of each attribute to be mutated
    # Visualization parameters
    'PLOT_REFRESH': 5,      # Number of generations between plot updates
    'POINTS_WEIGHT': 0.1,    # Weight for the points component in fitness function
    # Adaptive mutation parameters
    'STAGNATION_THRESHOLD': 10,  # Number of generations without improvement before increasing mutation
    'MUTATION_INCREASE': 0.1,   # How much to increase mutation rate
    'MAX_MUTATION_RATE': 0.3,    # Maximum mutation rate
}

# Clear any existing DEAP classes to avoid warnings
if 'FitnessMax' in globals():
    del creator.FitnessMax
if 'Individual' in globals():
    del creator.Individual

def generate_spiral(a, b, c, num_points):
    theta = np.linspace(0, c * np.pi, num_points)
    r = a + b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x -= x[0]
    y -= y[0]
    return x, y

def evaluate(individual):
    a, b, c, points = individual
    try:
        x, y = generate_spiral(a, b, c, int(points))
    except Exception:
        return 0,
    
    buckets = np.zeros(1000)
    x_min, x_max = min(x), max(x)
    
    if x_min == x_max:
        return 0,
    
    for i in range(len(x)):
        if isprime(i + 1):
            bucket_index = int((x[i] - x_min) / (x_max - x_min) * 999)
            buckets[bucket_index] += 1
    
    empty_buckets_percentage = np.sum(buckets == 0) / 1000
    
    # Add a component that rewards higher point counts
    points_score = (points - PARAMS['POINTS_MIN']) / (PARAMS['POINTS_MAX'] - PARAMS['POINTS_MIN'])
    
    # Combine the two components
    #fitness = (1 - PARAMS['POINTS_WEIGHT']) * empty_buckets_percentage + PARAMS['POINTS_WEIGHT'] * points_score
    fitness = empty_buckets_percentage * points_score
    return fitness,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_a", random.uniform, PARAMS['A_MIN'], PARAMS['A_MAX'])
toolbox.register("attr_b", random.uniform, PARAMS['B_MIN'], PARAMS['B_MAX'])
toolbox.register("attr_c", random.uniform, PARAMS['C_MIN'], PARAMS['C_MAX'])
toolbox.register("attr_points", random.randint, PARAMS['POINTS_MAX'] - 500, PARAMS['POINTS_MAX'])  # Start with high point counts
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_a, toolbox.attr_b, toolbox.attr_c, toolbox.attr_points), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                 low=[PARAMS['A_MIN'], PARAMS['B_MIN'], PARAMS['C_MIN'], PARAMS['POINTS_MIN']], 
                 up=[PARAMS['A_MAX'], PARAMS['B_MAX'], PARAMS['C_MAX'], PARAMS['POINTS_MAX']], 
                 eta=PARAMS['ETA'])
toolbox.register("mutate", tools.mutPolynomialBounded, 
                 low=[PARAMS['A_MIN'], PARAMS['B_MIN'], PARAMS['C_MIN'], PARAMS['POINTS_MIN']], 
                 up=[PARAMS['A_MAX'], PARAMS['B_MAX'], PARAMS['C_MAX'], PARAMS['POINTS_MAX']], 
                 eta=PARAMS['ETA'], indpb=PARAMS['INDPB'])
toolbox.register("select", tools.selTournament, tournsize=PARAMS['TOURNAMENT_SIZE'])

def main():
    pop = toolbox.population(n=PARAMS['POPULATION_SIZE'])
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plt.ion()
    
    max_fitness_history = []
    best_fitness = float('-inf')
    stagnation_counter = 0
    current_mutation_rate = PARAMS['P_MUTATION']
    
    for gen in range(PARAMS['MAX_GENERATIONS']):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=PARAMS['P_CROSSOVER'], mutpb=current_mutation_rate)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))
        
        record = stats.compile(pop)
        hof.update(pop)
        
        max_fitness_history.append(record['max'])
        
        # Check for improvement
        if record['max'] > best_fitness:
            best_fitness = record['max']
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Increase mutation rate if stagnant
        if stagnation_counter >= PARAMS['STAGNATION_THRESHOLD']:
            current_mutation_rate = min(current_mutation_rate + PARAMS['MUTATION_INCREASE'], PARAMS['MAX_MUTATION_RATE'])
            stagnation_counter = 0
            print(f"Increased mutation rate to {current_mutation_rate:.2f}")
        
        if gen % PARAMS['PLOT_REFRESH'] == 0:
            ax1.clear()
            ax1.plot(max_fitness_history, label='Max Fitness')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Best Fitness over Generations')
            ax1.legend()
            
            best = hof[0]
            x, y = generate_spiral(best[0], best[1], best[2], int(best[3]))
            
            ax2.clear()
            ax2.plot(x, y, label='Spiral')
            for i in range(len(x)):
                if isprime(i + 1):
                    ax2.plot(x[i], y[i], 'ro', markersize=3)
            ax2.set_aspect('equal')
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.axvline(0, color='black', linewidth=0.5)
            ax2.set_title(f"Best Spiral (Gen {gen})\na={best[0]:.2f}, b={best[1]:.4f}, c={best[2]:.2f}, Points={int(best[3])}\nFitness={best.fitness.values[0]:.4f}")
            
            plt.tight_layout()
            plt.pause(0.1)
        
        print(f"Generation {gen}: Best Fitness = {record['max']:.4f}, Mutation Rate = {current_mutation_rate:.2f}")
        print(f"Best: a = {best[0]:.2f}, b = {best[1]:.4f}, c = {best[2]:.2f}, Points = {int(best[3])}")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
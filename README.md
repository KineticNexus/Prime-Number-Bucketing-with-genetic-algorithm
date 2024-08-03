Explanation of the Code
This Python script uses the DEAP (Distributed Evolutionary Algorithms in Python) library to optimize the parameters of a spiral generation algorithm. The goal is to create a spiral with prime-numbered points that fall into the largest number of empty "buckets" or regions.
The hypothesis is that: if there are patterns, we can try and place all the prime numbers in a spiral, and change randomly the spiral shape (with a genetic algorithm) trying to align (vertically) as many prime numbers as possible in as little "buckets" in the X axis as poissible. If we can achieve to bucket most of the primes in a certain number of "buckets" these buckets might shed some light of the pilars of the prime numbers.

Here's a breakdown of the code:

Importing Libraries:

numpy for numerical operations.
matplotlib for plotting.
sympy to check for prime numbers.
deap for implementing the genetic algorithm.
random for generating random numbers.
Parameters:

PARAMS dictionary contains parameters for spiral generation, genetic algorithm settings, visualization options, and adaptive mutation control.
Helper Functions:

help(): Provides a detailed explanation of each parameter used in the script.
generate_spiral(a, b, c, num_points): Generates x and y coordinates for a spiral based on parameters a, b, c, and the number of points.
evaluate(individual): Evaluates the fitness of an individual by generating a spiral and calculating the percentage of empty buckets and a points score.
Genetic Algorithm Setup:

DEAP classes and toolbox are set up to define individuals, population, evaluation, mating, mutation, and selection processes.
Main Function:

main(): Runs the genetic algorithm, evolves the population over generations, adapts mutation rate if progress stagnates, and plots the best spiral and fitness progression.
![image](https://github.com/user-attachments/assets/4f1d6eac-805c-43b2-bc8f-376f2c3c0e4f)
![image](https://github.com/user-attachments/assets/64dab54a-a3dd-4f54-bacd-62f00dba5ef4)


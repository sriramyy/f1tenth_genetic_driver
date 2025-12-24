import random
import copy
import numpy as np
from .parameters import GeneticParameters

# lap class to store individual lap information
class Lap:
    def __init__(self, time, crash_count):
        self.time = time
        self.crash_count = crash_count

# genome represents a single candidate driver config
class Genome:
    def __init__(self, genome_id, params):
        self.id = genome_id
        self.params: GeneticParameters = params
        self.fitness_score = 0.0
        self.laps : Lap | list[Lap] = []

class GeneticML:
    def __init__(self, population_size=20, generations=50):
        self.population_size = population_size
        self.max_generations = generations
        self.current_generation = 1
        self.current_genome_index = 0
        self.elitism_percent = 0.2

        # overall best
        self.overall_best_genome : Genome | None = None

        # current gen ppool
        self.population = [] # population of genomes
        self.history = []    # history of our populations

        self.init_population()

    def init_population(self):
        """Creates the first generation"""
        print(f"Creating Initial Population (Gen {self.current_generation})")

        # make genome 0 a copy of default params
        default_genes = GeneticParameters()
        self.population.append(Genome(0, default_genes))

        # rest of genomes are mutated
        for i in range(1, self.population_size):
            p = GeneticParameters() # defaul params
            p = self.mutate(p) # mutate to create diversiy in first batch
            self.population.append(Genome(i,p))

    def get_next_genome_params(self):
        """returns GeneticParameter object for current test run"""
        if self.current_genome_index < len(self.population):
            return self.population[self.current_genome_index].params
        return self.population[0].params
    
    def report_lap_result(self, time, crash_count):
        """called after lap is finished to record it"""
        if self.current_genome_index >= len(self.population):
            return
        
        genome = self.population[self.current_genome_index]
        genome.laps.append(Lap(time, crash_count))

        # calc the fitness
        score = self.calculate_fitness(time, crash_count)
        genome.fitness_score = score

        # check and update overall best
        if self.overall_best_genome is None or score > self.overall_best_genome.fitness_score:
            self.overall_best_genome = copy.deepcopy(genome)
            print(f"\n 🟪🟪🟪 New Record: {time:.2f}s | Score: {score:.2f} ***")

        self.current_genome_index += 1

    def calculate_fitness(self, time, crash_count):
        """calc the fitness score of lap"""
        if time <= 0.1:
            return -1000.0
        
        speed_score = 1000.0 / time
        penalty = float(crash_count) * 500.0 # big penalty for hitting wall

        return speed_score - penalty
    
    def is_generation_complete(self):
        return self.current_genome_index >= len(self.population)
    
    def evolve_next_generation(self):
        """creates the next generation"""
        if self.current_generation >= self.max_generations:
            print("*** Evolution Complete ***")
            return
        
        print(f"Evolving to Generation {self.current_generation+1} ")

        # archive previous generation
        self.history.append(copy.deepcopy(self.population))

        # sort by descending fitness
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)

        next_gen = []

        # elitism, keep in top performers
        elite_count = int(self.population_size * self.elitism_percent)
        for i in range(elite_count):
            elite = copy.deepcopy(self.population[i])
            elite.id = len(next_gen)
            elite.laps = []
            elite.fitness_score = 0.0
            next_gen.append(elite)

        # fill in the rest by crossover
        while len(next_gen) < self.population_size:
            # pick 2 from top 50%
            limit = self.population_size//2
            p1 = self.population[random.randint(0, limit)].params
            p2 = self.population[random.randint(0, limit)].params

            # crossover
            child_params = self.crossover(p1, p2)

            # mutation
            child_params = self.mutate(child_params)

            # append
            next_gen.append(Genome(len(next_gen), child_params))
        
        # change everything to reflect new generation
        self.population = next_gen
        self.current_generation += 1
        self.current_genome_index = 0

    def crossover(self, p1, p2):
        """mixes (crosseS) genes from two parents"""
        child = copy.deepcopy(p1)

        # 50% for each gene from other parent
        attrs = vars(child)
        for attr in attrs:
            if random.random() > 0.5:
                setattr(child, attr, getattr(p2, attr))
        
        return child
    
    def mutate(self, p):
        """Randomly tweaks parameters to discover new behaviors."""
        rate = 0.3  # 30% chance for any gene to change
        
        # Float Mutations
        if random.random() < rate:
            p.MAX_LIDAR_DIST = np.clip(p.MAX_LIDAR_DIST + random.uniform(-0.5, 0.5), 2.0, 10.0)
        if random.random() < rate:
            p.STRAIGHT_SPEED = np.clip(p.STRAIGHT_SPEED + random.uniform(-0.5, 0.5), 1.0, 8.0)
        if random.random() < rate:
            p.CORNER_SPEED = np.clip(p.CORNER_SPEED + random.uniform(-0.3, 0.3), 0.5, p.STRAIGHT_SPEED)
        if random.random() < rate:
            p.CENTER_BIAS_ALPHA = np.clip(p.CENTER_BIAS_ALPHA + random.uniform(-0.1, 0.1), 0.0, 1.0)
        if random.random() < rate:
            p.STEER_SMOOTH_ALPHA = np.clip(p.STEER_SMOOTH_ALPHA + random.uniform(-0.1, 0.1), 0.0, 0.9)
            
        # Integer Mutations
        if random.random() < rate:
            p.BUBBLE_RADIUS = int(np.clip(p.BUBBLE_RADIUS + random.randint(-15, 15), 10, 250))
            
        return p
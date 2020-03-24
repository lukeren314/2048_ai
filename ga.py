from math import exp
from random import random, gauss, choice
SENSOR = 0
HIDDEN = 1
OUTPUT = 2


class GlobalInnovationCounter(object):
    counter = 0
    @staticmethod
    def get_innovation_number() -> int:
        GlobalInnovationCounter.counter += 1
        return GlobalInnovationCounter.counter - 1


class NodeGene:
    def __init__(self, id: int, node_type: int):
        self._id = id
        self._node_type = node_type

    def id(self) -> int:
        return self._id

    def node_type(self) -> int:
        return self._node_type


class ConnectGene:
    def __init__(self, in_node: int, out_node: int, weight: float, innovation: int, disabled: bool = False, recurrent: bool = False):
        self._in_node = in_node
        self._out_node = out_node
        self._weight = weight
        self._innovation = innovation
        self._disabled = disabled
        self._recurrent = recurrent

    def innovation(self) -> int:
        return self._innovation

    def copy(self) -> "ConnectGene":
        return ConnectGene(self._in_node, self._out_node, self._weight, self._innovation, self._disabled)

    def mutate(self, uniform_mutation_rate: float, replace_mutation_rate: float, replace_strength: float, mutation_strength: float, clamp_size: float) -> None:
        if random() < uniform_mutation_rate:
            self._weight = max(-clamp_size, min(clamp_size,
                                                self._weight+gauss(0.0, mutation_strength)))
        elif random() < replace_mutation_rate:
            self._weight = gauss(0.0, replace_strength)

    def disable(self) -> None:
        self._disabled = True


class Genome:
    def __init__(self, input_size: int, output_size: int):
        self._input_size = input_size
        self._output_size = output_size
        self._initialize_genes()
        self._add_connection()
        self._build_phenotype()
        self._last_created_connections = []
        self._fitness = 0.0
        self._adjusted_fitness = 0.0

    def feedforward(self, inputs: [float]) -> [float]:
        pass

    def fitness(self) -> float:
        return self._fitness

    def adjusted_fitness(self) -> float:
        return self._adjusted_fitness

    def assign_adjusted_fitness(self, val: float) -> None:
        self._adjusted_fitness = val

    def _initialize_genes(self) -> None:
        for i in range(self._input_size):
            self._node_genes[i] = NodeGene(i, SENSOR)
        for i in range(self._input_size+1, self._input_size + 1 + self._output_size):
            self._node_genes[i] = NodeGene(i, OUTPUT)
        self._connection_genes = {}

    def _build_phenotype(self) -> None:
        pass

    def _add_connection(self, replace_strength: float = 2.0, recurrent_connection_rate: float = 0.0, ) -> None:
        if random() < recurrent_connection_rate:
            in_node = choice(
                [node for node in self._node_genes if self._node_genes[node].node_type() != INPUT])
            out_node = choice(self._get_lower_nodes(in_node))
        else:
            in_node = choice(
                [node for node in self._node_genes if self._node_genes[node].node_type() != OUTPUT])
            out_node = choice(
                [node for node in self._node_genes if self._node_genes[node].node_type() != SENSOR])
            weight = gauss(0.0, replace_strength)
            disabled = False
            recurrent = False
            connection = self._create_connection(
                in_node, out_node, weight, disabled, recurrent)
            self._connection_genes[connection.innnovation()]

    def _create_connection(self, in_node: int, out_node: int, weight: float, disabled: bool, recurrent: bool) -> ConnectGene:
        connection = ConnectGene(
            in_node, out_node, weight, GlobalInnovationCounter.get_innovation_number(), disabled, recurrent)
        self._last_created_connections.append(connection)
        return connection

    def _get_lower_nodes(self, in_node: int) -> None:
        pass

    def _add_node(self) -> None:
        # disables old connection, replaces with node
        # incoming weight -> 1
        # outgoing weight -> weight of old connection
        old_connection_innovation = choice(
            [connection for connection in self._connection_genes])
        self._connection_genes[old_connection_innovation].disable()

    def _create_node(self, node_type: int) -> NodeGene:
        pass

    def _calculate_compatibility_distance(self, other_genome: Genome, c1: float, c2: float, c3: float) -> float:
        # linear combination of number of excess/disjoint genes
        # aka "delta"
        # return c1 (importance) * E (number of excess) / N
        # (max(num genes in genome1, num genes in genome2)) +
        # c2 * D (number of disjoint) / N +
        # c3 * W (avereage weight differences of matching genes)
        # N set to one if MAX is less than 20
        pass

    def _activate(self, x: float) -> float:
        # modified sigmoidal transfer function
        return 1/(1+exp(-4.9*x))


class Species:
    def __init__(self):
        self._genomes = []

    def genomes(self) -> [Genome]:
        return self._genomes

    def population_size(self) -> int:
        return len(self._genomes)

    def calculate_adjusted_fitnesses(self) -> None:
        for genome in self._genomes:
            genome.assign_adjusted_fitness(genome.fitness()/len(self._genomes))

    def total_adjusted_fitness(self) -> float:
        return sum([genome.adjusted_fitness for genome in self._genomes])

    def eliminate_lowest(self, elimination_rate: float) -> None:
        self._genomes = [genome for i, genome in enumerate(sorted(
            self._genomes, key=lambda g: g.adjusted_fitness())) if i / len(self._genomes) > elimination_rate]

    def pick_highest_fitness(self) -> Genome:
        best = self._genomes[0]
        for genome in self._genomes:
            if genome.fitness() > best.fitness():
                best = genome
        return best

    def get_first_genome(self) -> Genome:
        return self._genomes[0]


class NEAT:
    def __init__(self, input_size: int, output_size: int,
                 population_size: int = 150,
                 delta_threshold: float = 3.0,
                 # increase (to 4.0) for higher c3 (larger populations)
                 elimination_rate: float = 0.5,
                 lenient_generations: int = 15,
                 minimum_unchanged_champion_size: int = 5,
                 connection_mutation_rate: float = 0.8,
                 uniform_mutation_rate: float = 0.9,
                 replace_mutation_rate: float = 0.1,
                 mutation_strength: float = 1.5,
                 replace_strength: float = 2.0,
                 clamp_size: float = 5.0,
                 inherit_disabled_rate: float = 0.75,
                 mutation_without_crossover_rate: float = 0.25,
                 interspecies_mating_rate: float = 0.001,
                 new_node_rate: float = 0.03,
                 new_connection_rate: float = 0.05,
                 # increase (to 0.3) for greater topological diversity (larger populations)
                 recurrent_connection_rate: float = 0.05,
                 c1: float = 1.0,
                 c2: float = 1.0,
                 c3: float = 0.4):  # increase c3 (to 3.0) for finer distinction
                 # between species based on weights (usually for larger
                 # populations)
        self._input_size = input_size
        self._output_size = output_size
        self._population_size = population_size
        self._delta_threshold = delta_threshold
        self._elimination_rate = elimination_rate
        self._lenient_generations = lenient_generations
        self._minimum_unchanged_champion_size = minimum_unchanged_champion_size
        self._connection_mutation_rate = connection_mutation_rate
        self._uniform_mutation_rate = uniform_mutation_rate
        self._replace_mutation_rate = replace_mutation_rate
        self._mutation_strength = mutation_strength
        self._replace_strength = replace_strength
        self._clamp_size = clamp_size
        self._inherit_disabled_rate = inherit_disabled_rate
        self._mutation_without_crossover_rate = mutation_without_crossover_rate
        self._interspecies_mating_rate = interspecies_mating_rate
        self._new_node_rate = new_node_rate
        self._new_connection_rate = new_connection_rate
        self._recurrent_connection_rate = recurrent_connection_rate
        self._c1 = c1
        self._c2 = c2
        self._c3 = c3
        self._species = []

    def next_generation(self) -> None:
        # calculate fitnesses?
        # kill worst genomes of each species
        # repopulate the species by mating within species
        generation_fitness = 0.0
        for species in self._species:
            species.calculate_adjusted_fitnesses()
            species.eliminate_lowest(self._elimination_rate)
            generation_fitness += species.total_adjusted_fitness()

        i = 0
        while i < self._population_size:
            i += 1

    def _mate(self, genome1: Genome, genome2: Genome) -> None:
        # matches up connections with same innovation number
        # and takes the disjoint/excess genes of the
        # parent that is more fit. Matching genes are
        # inherited randomly. If the fitnesses are equal,
        # then disjoint/excess are also randomly assigned.
        pass

    def _speciate(self) -> None:
        # get a list of random genomes from each of the last species
        # and go through every genome in the current genome.
        # put the current genome into the first genome it matches with,
        # and if it doesn't, create a new species for it.
        pass

    def _calcualte_adjusted_fitness(self, genome: Genome, species: [Genome]) -> float:
        # calculates an adjusted fitness based on the species it came from
        # return f (raw fitness of the genome) /
        # sum(sh(delta(genome))) (sh = 0 if delta(genome) is above the
        # delta threshold else 1).
        # OR f / n (number of genomes in the species)
        pass

    def _match_same_innovations(self) -> None:
        # goes through each connection in the generation
        # and if they are the same, give them the
        # same innovation number.
        pass


if __name__ == "__main__":
    print(GlobalInnovationCounter.get_innovation_number())
    print(GlobalInnovationCounter.get_innovation_number())

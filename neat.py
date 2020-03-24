from math import exp
from random import random, gauss, choice

INPUT = 0
HIDDEN = 1
OUTPUT = 2


class GlobalInnovationCounter(object):
    counter = 0
    @staticmethod
    def get_innovation_number() -> int:
        GlobalInnovationCounter.counter += 1
        return GlobalInnovationCounter.counter - 1


class ConnectionGene:
    def __init__(self, innovation_number: int, in_node: int, out_node: int, weight: float, disabled: bool, recurrent: bool):
        self.innovation_number = innovation_number
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.disabled = disabled
        self.recurrent = recurrent

    def mutate(self, perturb_rate: float, perturb_sigma: float, replace_rate: float, replace_sigma: float, clamp_size: float) -> None:
        if random() < perturb_rate:
            self.weight = max(-clamp_size, min(clamp_size,
                                               self.weight+gauss(0.0, perturb_sigma)))
        elif random() < replace_rate:
            self.weight = gauss(0.0, replace_sigma)

    def disable(self) -> None:
        self.disabled = False


class NodeGene:
    def __init__(self, id_: int, node_type: int):
        self.id = id_
        self.node_type = node_type
        self.out_connections = []
        self.in_connections = []
        self.height = 0
        self.reset()

    def reset(self) -> None:
        self.fired = False
        self.val = 0

    def reset_connections(self) -> None:
        self.in_connections = []
        self.out_connections = []

    def transfer(self, val: float) -> None:
        self.val += val

    def copy(self) -> "NodeGene":
        new_node = NodeGene(self.id, self.node_type)
        new_node.in_connections = [
            connection for connection in self.in_connections]
        new_node.out_connections = [
            connection for connection in self.out_connections]


class Genome:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.node_genes = {i: NodeGene(
            i, INPUT if i < input_size else OUTPUT) for i in range(input_size+output_size)}
        self.input_nodes = [i for i in range(input_size)]
        self.output_nodes = [i for i in range(
            input_size, input_size+output_size)]
        self.connection_genes = {}
        self.last_created_connections = []
        self.fitness = 0.0
        self.adjusted_fitness = 0.0

        self._build_network()

    def feedforward(self, inputs: [float]) -> [float]:
        assert self.input_size == len(inputs)
        self._clear_network()
        for i in self.input_nodes:
            self.node_genes[i].transfer(inputs[i])
        layer = self.input_nodes
        while len(layer) > 0:
            next_layer = []
            for node in layer:
                for connection in self.node_genes[node].out_connections:
                    if connection.out_node not in next_layer:
                        next_layer.append(connection.out_node)
                if self._full_node(node):
                    self._fire_node(node)
        return [self.output_nodes[node].val for node in self.output_nodes]

    def _full_node(self, node_id: int) -> bool:
        for in_connection in self.node_genes[node_id].in_connections:
            if not self.node_genes[self.connection_genes[in_connection].in_node].fired:
                return False
        return True

    def _fire_node(self, node_id: int) -> None:
        for out_connection in self.node_genes[node_id].out_connections:
            self.node_genes[self.connection_genes[out_connection].out_node].transfer(
                self._activate(self.connection_genes[out_connection]).weight*self.node_genes[node_id].val)
        self.node_genes[node_id].fired = True

    def mutate(self, add_connection_rate: float, add_node_rate: float, connection_mutation_rate: float) -> None:
        if random() < add_connection_rate:
            self.add_random_connection()
        if random() < add_node_rate:
            self.add_random_node()
        for connection in self.connection_genes:
            if random() < connection_mutation_rate:
                self.connection_genes[connection].mutate()

    def calculate_compatibility_distance(self, genome2: Genome, c1: float, c2: float, c3: float) -> float:
        m1 = max(self.connection_genes)
        m2 = max(genome2.connection_genes)
        e = 0
        d = 0
        for i in range(max(m1, m2)):
            if i > m1 or i > m2:
                e += 1
            elif i in self.connection_genes and i not in genome2.connection_genes or i in genome2 and i not in self.connection_genes:
                d += 1
        w = 0
        c = 0
        for i in self.connection_genes:
            if i in genome2.connection_genes:
                w += abs(self.connection_genes[i].weight -
                         genome2.connection_genes[i].weight)
                c += 1
        w /= c

        n = max(len(self.node_genes)+len(self.connection_genes),
                len(genome2.node_genes, genome2.connection_genes))
        n = 1 if n < 20 else n
        return c1 * e / n + c2 * d / n + c3 * w

    def add_random_connection(self, recurrent_chance: float = 0.0) -> None:
        in_node = choice(
            [node for node in self.node_genes if self.node_genes[node].node_type() != OUTPUT])

        recurrent = False
        weight = gauss(0.0, 1.0)
        disabled = False

        if random() < recurrent_chance:
            out_node = self._get_random_lower_node(in_node)
            recurrent = True
        else:
            out_node = self._get_random_higher_node(in_node)

        self._add_connection(self._create_connection(
            in_node, out_node, weight, disabled, recurrent))

    def add_random_node(self) -> None:
        old_connection = self.connection_genes[choice(
            [connection for connection in self.connection_genes])]
        old_connection.disable()
        new_node = self._create_node()
        self._add_node(new_node)

        self._add_connection(self._create_connection(
            old_connection.in_node, new_node, 1.0))
        self._add_connection(self._create_connection(
            new_node, old_connection.out_node, old_connection.weight))

    def copy(self) -> Genome:
        new_copy = Genome(self.input_size, self.output_size)
        new_copy._copy_node_genes(self.node_genes)
        new_copy._copy_connection_genes(self.connection_genes)

    def _copy_node_genes(self, node_genes2: {int: NodeGene}) -> None:
        self.node_genes = {
            node_genes2[node].id: node_genes2[node].copy() for node in node_genes2}

    def _copy_connection_genes(self, connection_genes2: {int: ConnectionGene}) -> None:
        self.connection_genes = {connection_genes2[connection].innovation: connection_genes2[connection].copy(
        ) for connection in connection_genes2}

    def cross_over(self, genome2: Genome) -> Genome:

        return

    def _build_network(self) -> None:
        self._reset_network()
        for connection in self.connection_genes:
            self.node_genes[self.connection_genes[connection].in_node].out_connections.append(
                connection)
            self.node_genes[self.connection_genes[connection].out_node].in_nodes.append(
                connection)
        layer = []
        for node in self.input_nodes:
            self.node_genes[node].height = 0
            layer.append(self.node_genes[node].out_nodes)
        while len(layer) > 0:
            next_layer = []
            for node in next_layer:
                node.height = max([self.node_genes[self.connection_genes[in_connection].in_node].height
                                   for in_connection in self.node_genes[node].in_connections]) + 1
                next_layer += [self.connection_genes[out_connection]
                               .out_node for out_connection in self.node_genes[node].out_connections]
            layer = next_layer

    def _reset_network(self) -> None:
        for node in self.node_genes:
            self.node_genes[node].reset_connections()

    def _clear_network(self) -> None:
        for node in self.node_genes:
            self.node_genes[node].reset()

    def _add_connection(self, connection: ConnectionGene) -> None:
        self.last_created_connections.append(connection)
        self.connection_genes[connection.innovation_number] = connection

    def _get_random_lower_node(self, in_node: int) -> int:
        return choice([node for node in self.node_genes if self.node_genes[node].height <= self.node_genes[in_node].height])

    def _get_random_higher_node(self, in_node: int) -> int:
        return choice([node for node in self.node_genes if self.node_genes[node].height > self.node_genes[in_node].height])

    def _create_connection(self, in_node: int, out_node: int, weight: float, disabled: bool = False, recurrent: bool = False) -> ConnectionGene:
        return ConnectionGene(
            in_node, out_node, weight, GlobalInnovationCounter.get_innovation_number(), disabled, recurrent)

    def _add_node(self, node: NodeGene) -> None:
        self.node_genes[node.id] = node

    def _create_node(self) -> NodeGene:
        return NodeGene(max([node for node in self.node_genes]) + 1, HIDDEN)

    def _activate(self, x: float) -> float:
        # modified sigmoidal transfer function
        return 1/(1+exp(-4.9*x))


class Species:
    def __init__(self):
        self.genomes = []
        self.next_genomes = []

    def add_genome(self, genome: Genome) -> None:
        self.next_genomes.append(genome)

    def next_generation(self) -> None:
        self.genomes = self.next_genomes
        self.next_genomes = []

    def population_size(self) -> int:
        return len(self.genomes)

    def calculate_adjusted_fitnesses(self) -> None:
        for genome in self.genomes:
            genome.adjusted_fitness = genome.fitness/len(self.genomes)

    def total_adjusted_fitness(self) -> float:
        return sum([genome.adjusted_fitness for genome in self.genomes])

    def eliminate_lowest_fitness(self, elimination_rate: float) -> None:
        self.genomes = [genome for i, genome in enumerate(sorted(
            self.genomes, key=lambda g: g.adjusted_fitness())) if i / len(self.genomes) > elimination_rate]

    def pick_highest_fitness(self) -> Genome:
        return [genome for i, genome in enumerate(sorted(
            self.genomes, key=lambda g: g.adjusted_fitness()))][-1]

    def get_random_genome(self) -> Genome:
        return choice(self.genomes)

    def get_two_random_genomes(self) -> Genome:
        genome1 = choice(self.genomes)
        genome2 = choice(
            [genome for genome in self.genomes if genome is not genome1])
        return genome1, genome2


class NEAT:
    def __init__(self, input_size: int, output_size: int,
                 population_size: int = 150,
                 elimination_rate: float = 0.5,
                 mutation_without_crossover_rate: float = 0.25):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.elimination_rate = elimination_rate
        self.mutation_without_crossover_rate = mutation_without_crossover_rate
        self.genomes = []
        self.species = []
        self._build_initial_population()

    def next_generation(self) -> None:
        generation_fitness = 0.0
        for species in self.species:
            species.calculate_adjusted_fitnesses()
            species.eliminate_lowest(self.elimination_rate)
            generation_fitness += species.total_adjusted_fitness()
        while len(self.genomes) < self.population_size:
            r = random()
            i = 0
            while r > 0:
                r -= self.species[i].total_adjusted_fitness() / \
                    generation_fitness
                i += 1
            i -= 1

            if random() < self.mutation_without_crossover_rate:
                new_genome = self.species[i].get_random_genome()
                new_genome.mutate()
                self._add_genome(new_genome)
            else:
                genome1, genome2 = self.species[i].get_two_random_genomes()
                # crossover and add genome
                new_genome = genome1.cross_over(genome2)
                new_genome.mutate()
                self._add_genome(new_genome)
                # repopulate by mating within species
                # place into new species

    def _build_initial_population(self) -> None:
        self.species.append(Species())
        for _ in range(self.population_size):
            genome = Genome(self.input_size, self.output_size)
            genome.add_random_connection()
            self.genomes.append(genome)
            self.species[0].add_genome(genome)
        self.species[0].next_generation()

    def _add_genome(self, genome: Genome) -> None:
        self.genomes.append(genome)
        # speciate genome

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

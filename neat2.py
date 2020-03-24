from math import exp
from random import random, gauss, choice

INPUT = 0
HIDDEN = 1
OUTPUT = 2

# cleanup
# extra features


class GlobalInnovationCounter(object):
    counter = 0
    @staticmethod
    def get_innovation_number() -> int:
        GlobalInnovationCounter.counter += 1
        return GlobalInnovationCounter.counter - 1


class ConnectionGene:
    def __init__(self, innovation_number: int, in_node: int, out_node: int, weight: float, disabled: bool):
        self.innovation_number = innovation_number
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.disabled = disabled

    def copy(self) -> "ConnectionGene":
        return ConnectionGene(self.innovation_number, self.in_node, self.out_node, self.weight, self.disabled)

    def mutate(self, perturb_rate: float, perturb_sigma: float, replace_rate: float, replace_sigma: float, clamp_size: float) -> None:
        if random() < perturb_rate:
            self.weight = max(-clamp_size, min(clamp_size,
                                               self.weight+gauss(0.0, perturb_sigma)))
        elif random() < replace_rate:
            self.weight = gauss(0.0, replace_sigma)


class NodeGene:
    def __init__(self, id_: int, node_type: int):
        self.id = id_
        self.node_type = node_type

    def copy(self) -> "NodeGene":
        new_node = NodeGene(self.id, self.node_type)
        new_node.in_connections = [
            connection for connection in self.in_connections]
        new_node.out_connections = [
            connection for connection in self.out_connections]


class Genome:
    def __init__(self, input_size: int, output_size: int) -> "Genome":
        self.input_size = input_size
        self.output_size = output_size
        self.node_genes = {}
        self.connection_genes = {}

        self.fitness = 0.0
        self.adjusted_fitness = 0.0

    def initialize(self) -> None:
        self.node_genes = {i: NodeGene(i, INPUT if i < self.input_size else OUTPUT) for i in range(
            self.input_size+self.output_size)}
        self.add_connection(self._create_connection(
            choice(self.input_nodes()), choice(self.output_nodes()), gauss(0.0, 1.0), random() < 0.5))
        self.build_network()

    def build_network(self) -> None:
        self.network = Network(self)

    def input_nodes(self) -> [int]:
        return [i for i in range(self.input_size)]

    def output_nodes(self) -> [int]:
        return [i for i in range(self.input_size, self.input_size+self.output_size)]

    def copy(self) -> "Genome":
        new_genome = Genome(self.input_size, self.output_size)
        new_genome.node_genes = {node_id: self.node_genes[node_id].copy(
        ) for node_id in self.node_genes}
        new_genome.connection_genes = {connection_innovation: self.connection_genes[connection_innovation].copy(
        ) for connection_innovation in self.connection_genes}
        return new_genome

    def add_connection(self, connection: ConnectionGene) -> None:
        self.connection_genes[connection.innovation_number] = connection

    def add_node(self, node: NodeGene) -> None:
        self.node_genes[node.id] = node

    def cross_over(self, genome2: genome2, inherit_disabled_rate: float) -> Genome:
        # matches up connections with same innovation number
        # and takes the disjoint/excess genes of the
        # parent that is more fit. Matching genes are
        # inherited randomly. If the fitnesses are equal,
        # then disjoint/excess are also randomly assigned.
        fitter_parent, weaker_parent = self, genome2 if self.adjusted_fitness > genome2.adjusted_fitness else genome2, self

        if fitter_parent.adjusted_fitness != weaker_parent.adjusted_fitness:
            new_genome = fitter_parent.copy()
            for i in fitter_parent.connection_genes:
                if i in weaker_parent.conection_genes:
                    new_genome._inherit_merged_connection(
                        fitter_parent.connection_genes[i], weaker_parent.connection_genes[i], inherit_disabled_rate)
        else:
            new_genome = Genome(self.input_size, self.output_size)
            for i in fitter_parent.connection_genes.keys() | weaker_parent.keys():
                if i in fitter_parent.connection_genes and i in weaker_parent.conection_genes:
                    new_genome._inherit_merged_connection(
                        fitter_parent.connection_genes[i], weaker_parent.connection_genes[i], inherit_disabled_rate)
                elif i in fitter_parent.connection_genes:
                    new_genome._inherit_connection(
                        fitter_parent.connection_genes[i], inherit_disabled_rate)

                else:
                    new_genome._inherit_connection(
                        weaker_parent.connection_genes[i], inherit_disabled_rate)

        return new_genome

    def mutate(self, new_connection_rate: float, new_node_rate: float, connection_mutation_rate: float, recurrent_connection_rate: float, perturb_rate: float, perturb_sigma: float, replace_rate: float, replace_sigma: float, clamp_size: float) -> None:
        if random() < new_connection_rate:
            self._add_random_connection(recurrent_connection_rate)
        if random() < new_node_rate:
            self._add_random_node()
        for connection in self.connection_genes:
            if random() < connection_mutation_rate:
                self.connection_genes[connection].mutate(
                    perturb_rate, perturb_sigma, replace_rate, replace_sigma, clamp_size)

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

    def _inherit_merged_connection(self, connection1: ConnectionGene, connection2: ConnectionGene, inherit_disabled_rate: float) -> None:
        new_connection = self._pick_connection(
            connection1, connection2).copy()
        new_connection.disabled = self._inherit_disabled(
            connection1.disabled or connection2.disabled, inherit_disabled_rate)
        self.add_connection(new_connection)

    def _inherit_connection(self, connection: ConnectionGene, inherit_disabled_rate: float) -> None:
        if random() < 0.5:
            new_connection = connection.copy()
            new_connection.disabled = self._inherit_disabled(
                connection.disabled, inherit_disabled_rate)
            self.add_connection(new_connection)

    def _pick_connection(self, connection1: ConnectionGene, connection2: ConnectionGene) -> ConnectionGene:
        return connection1 if random() < 0.5 else connection2

    def _inherit_disabled(self, parent_disabled: bool, inherit_disabled_rate: float) -> bool:
        if parent_disabled and random() < inherit_disabled_rate:
            return True
        return False

    def _add_random_connection(self, recurrent_connection_rate: float) -> None:
        in_node = choice(
            [node for node in self.node_genes if self.node_genes[node].node_type != OUTPUT])
        recurrent = False
        weight = gauss(0.0, 1.0)
        disabled = False
        if random() < recurrent_connection_rate:
            out_node = self.network.get_random_lower_node(in_node)
            recurrent = True
        else:
            out_node = self.network.get_random_higher_node(in_node)
        # do something about recurrent connections
        self.add_connection(self._create_connection(
            in_node, out_node, weight, disabled))

    def _add_random_node(self) -> None:
        # check if not recurrent
        old_connection = self.connection_genes[choice(
            [connection for connection in self.connection_genes])]
        old_connection.disable()
        new_node = self._create_node()
        self.add_node(new_node)

        self.add_connection(self._create_connection(
            old_connection.in_node, new_node, 1.0))
        self.add_connection(self._create_connection(
            new_node, old_connection.out_node, old_connection.weight))

    def _create_connection(self, in_node: int, out_node: int, weight: float, disabled: bool = False) -> ConnectionGene:
        return ConnectionGene(
            in_node, out_node, weight, GlobalInnovationCounter.get_innovation_number(), disabled)

    def _create_node(self) -> NodeGene:
        return NodeGene(max([node for node in self.node_genes]) + 1, HIDDEN)


class Node:
    def __init__(self, id_: int):
        self.id = id_
        self.height = 0
        self.out_connections = {}
        self.in_nodes = []
        self.val = 0.0
        self.next_val = 0.0
        self.reset()

    def add_connection(self, connection_gene: ConnectionGene) -> None:
        self.out_connections[connection_gene.out_node] = connection_gene.weight

    def add_in_node(self, node_id: int) -> None:
        self.in_nodes.append(node_id)

    def reset(self) -> None:
        self.fired = False
        self.val = self.next_val
        self.next_val = 0

    def transfer(self, val: float) -> None:
        self.val += val

    def store(self, val: float) -> None:
        self.next_val += val


class Network:
    def __init__(self, genome: Genome):
        self.input_size = genome.input_size
        self.output_size = genome.output_size
        self._build_nodes(genome)
        self._build_connections(genome)
        self._build_heights()

    def feedforward(self, inputs: [float]) -> [float]:
        assert self.input_size == len(inputs)
        self._reset_network()
        for i in self.input_nodes:
            self.nodes[i].transfer(inputs[i])

        layer = self.input_nodes
        while len(layer) > 0:
            next_layer = []
            for node in layer:
                for out_node in self.nodes[node].out_connections:
                    if out_node not in next_layer:
                        next_layer.append(out_node)
                if self._ready_node(node):
                    self._fire_node(node)
        return [self.nodes[node].val for node in self.output_nodes]

    def get_random_lower_node(self, in_node: int) -> int:
        return choice([node for node in self.nodes if self.nodes[node].height <= self.nodes[in_node].height])

    def get_random_higher_node(self, in_node: int) -> int:
        return choice([node for node in self.nodes if self.nodes[node].height > self.nodes[in_node].height])

    def _build_nodes(self, genome: Genome) -> None:
        self.nodes = {node_id: Node(node_id)
                      for node_id in genome.node_genes}
        self.input_nodes = genome.input_nodes()
        self.output_nodes = genome.output_nodes()

    def _build_connections(self, genome: Genome) -> None:
        self.connections = []
        for connection_innovation in genome.connection_genes:
            if not genome.connection_genes.disabled:
                self.nodes[genome.connection_genes[connection_innovation].in_node].add_connection(
                    genome.connection_genes[connection_innovation])
                self.nodes[genome.connection_genes[connection_innovation].out_node].add_in_node(
                    genome.connection_genes[connection_innovation].in_node)

    def _build_heights(self) -> None:
        layer = []
        for node in self.input_nodes:
            layer += [node_id for node_id in self.nodes[node].out_connections]
        while len(layer) > 0:
            next_layer = []
            for node in next_layer:
                node.height = max(
                    [self.nodes[in_node].height for in_node in self.nodes[node].in_nodes]) + 1
                next_layer += [out_node for out_node in self.nodes[node].out_connections]
            layer = next_layer

    def _ready_node(self, node_id: int) -> bool:
        for in_node in self.nodes[node_id].in_nodes:
            if not self.nodes[in_node].fired:
                return False
        return True

    def _fire_node(self, node_id: int) -> None:
        for out_node in self.nodes[node_id].out_connections:
            if self.nodes[out_node].fired:
                # recurrent connection
                self.nodes[out_node].store(self._activate(
                    self.nodes[node_id].out_connections[out_node]).weight*self.nodes[node_id].val)
            else:
                self.nodes[out_node].transfer(self._activate(
                    self.nodes[node_id].out_connections[out_node]).weight*self.nodes[node_id].val)
        self.nodes[node_id].fired = True

    def _reset_network(self) -> None:
        for node_id in self.nodes:
            self.nodes[node_id].reset()

    def _activate(self, x: float) -> float:
        # modified sigmoidal transfer function
        return 1/(1+exp(-4.9*x))


class Species:
    def __init__(self):
        self.genomes = []
        self.next_genomes = []

    def match(self, genome: Genome, c1: float, c2: float, c3: float, delta_threshold: float) -> bool:
        if self.get_random_genome().calculate_compatibility_distance(genome, c1, c2, c3) < delta_threshold:
            return True
        return False

    def queue_genome(self, genome: Genome) -> None:
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
                 mutation_without_crossover_rate: float = 0.25,
                 inherit_disabled_rate: float = 0.75,
                 add_connection_rate: float = 0.3,
                 add_node_rate: float = 0.03,
                 connection_mutation_rate: float = 0.8,
                 c1: float = 1.0,
                 c2: float = 1.0,
                 c3: float = 0.4,
                 recurrent_rate: float = 0.05,
                 delta_threshold: float = 3.0):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.elimination_rate = elimination_rate
        self.mutation_without_crossover_rate = mutation_without_crossover_rate
        self.inherit_disabled_rate = inherit_disabled_rate
        self.add_connection_rate = add_connection_rate
        self.add_node_rate = add_node_rate
        self.connection_mutation_rate = connection_mutation_rate
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.recurrent_rate = recurrent_rate
        self.delta_threshold = delta_threshold

        self.genomes = []
        self.networks = []
        self.next_genomes = []
        self.species = []
        self._build_initial_population()

    # get some way for phenotypes to give their fitness to the genome

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
            else:
                genome1, genome2 = self.species[i].get_two_random_genomes()
                new_genome = genome1.cross_over(genome2)

            new_genome.mutate()
            self._speciate_genome(new_genome)
            # repopulate by mating within species
            # place into new species

    def _build_initial_population(self) -> None:
        self.species.append(Species())
        for _ in range(self.population_size):
            genome = Genome(self.input_size, self.output_size)
            genome.initialize()
            # add random connection
            self.genomes.append(genome)
            self.networks.append(Network(genome))
            self.species[0].queue_genome(genome)
        self.species[0].next_generation()

    def _speciate_genome(self, genome: Genome) -> None:
        # speciate genome
        # get a list of random genomes from each of the last species
        # and go through every genome in the current genome.
        # put the current genome into the first genome it matches with,
        # and if it doesn't, create a new species for it.
        for species in self.species:
            if species.match(genome, self.c1, self.c2, self.c3, self.delta_threshold):
                species.queue_genome(genome)
        self._add_genome(genome)

    def _add_genome(self, genome: Genome) -> None:
        self.genomes.append(genome)

    def _speciate(self) -> None:

        pass

    def _match_same_innovations(self) -> None:
        # goes through each connection in the generation
        # and if they are the same, give them the
        # same innovation number.
        pass

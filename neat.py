from math import exp
from random import random, gauss, choice

INPUT = 0
HIDDEN = 1
OUTPUT = 2


class Connection:
    def __init__(self, innovation: int, in_node: int, out_node: int, weight: float, disabled: bool, recurrent: bool):
        self.innovation = innovation
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.disabled = disabled
        self.recurrent = recurrent

    def as_dict(self) -> dict:
        return {
            'innovation': self.innovation,
            'in_node': self.in_node,
            'out_node': self.out_node,
            'weight': self.weight,
            'disabled': self.disabled,
            'recurrent': self.recurrent
        }

    def copy(self) -> 'Connection':
        return Connection(self.innovation, self.in_node, self.out_node, self.weight, self.disabled, self.recurrent)


class Node:
    def __init__(self, id_: int, node_type: int):
        self.id = id_
        self.node_type = node_type
        self.height = 0 if node_type == INPUT else 1
        self.val = 0.0
        self.next_val = 0.0
        self.fired = False

        self.out_connections = {}
        self.in_nodes = []

    def copy(self) -> 'Node':
        new_node = Node(self.id, self.node_type)
        new_node.height = self.height

        for out in self.out_connections:
            new_node.out_connections[out] = self.out_connections[out]
        for in_node in self.in_nodes:
            new_node.in_nodes.append(in_node)
        return new_node

    def add_connection(self, connection: Connection) -> None:
        self.out_connections[connection.out_node] = connection.weight

    def add_in_node(self, node_id: int) -> None:
        self.in_nodes.append(node_id)

    def reset(self) -> None:
        self.val = self.next_val
        self.next_val = 0
        self.fired = False

    def transfer(self, val: float) -> None:
        self.val += val

    def store(self, val: float) -> None:
        self.next_val += val


class Genome:
    def __init__(self, input_size: int, output_size: int, connections: [Connection]):
        self.input_size = input_size
        self.output_size = output_size
        self.connections = connections

        self.fitness = 0.0
        self.adjusted_fitness = 0.0

        self.nodes = {}
        self.input_nodes = []
        self.output_nodes = []
        self._initialize_nodes()
        self._initialize_node_connections()

        self.max_height = 0
        self._initialize_heights()

        self.network = {}

    @staticmethod
    def from_list(input_size: int, output_size: int, connections: [dict]) -> 'Genome':
        connection_genes = []
        for c in connections:
            connection_genes.append(Connection(
                c['innovation'], c['in_node'], c['out_node'], c['weight'], c['disabled'], c['recurrent']))
        new_genome = Genome(input_size, output_size, connection_genes)
        new_genome.initialize_network()
        return new_genome

    def as_list(self) -> list:
        return [connection.as_dict() for connection in self.connections]

    def copy(self) -> 'Genome':
        new_connections = [connection.copy()
                           for connection in self.connections]
        new_genome = Genome(self.input_size, self.output_size, new_connections)
        for node in self.nodes:
            new_genome.nodes[node] = self.nodes[node].copy()
        new_genome.input_nodes = [i for i in self.input_nodes]
        new_genome.output_nodes = [o for o in self.output_nodes]
        new_genome.max_height = self.max_height
        new_genome.network = {}
        for node in self.network:
            new_genome.network[node] = self.network[node]
        return new_genome

    def feedforward(self, inputs: [float]) -> [float]:
        assert self.input_size == len(inputs)
        self._reset_network()
        for i in self.input_nodes:
            self.nodes[i].transfer(inputs[i])
        for layer_height in self.network:
            for node_id in self.network[layer_height]:
                self._fire_node(node_id)
        return [self.nodes[output_node].val for output_node in self.output_nodes]

    def innovations(self) -> [int]:
        return [connection.innovation for connection in self.connections]

    def get_connection(self, innovation: int) -> Connection:
        for connection in self.connections:
            if connection.innovation == innovation:
                return connection

    def find_connection(self, in_node: int, out_node: int) -> Connection:
        for connection in self.connections:
            if connection.in_node == in_node and connection.out_node == out_node:
                return Connection

    def modify_height(self, node_id: int) -> None:
        node = self.nodes[node_id]
        old_height = node.height
        node.height = max([self.nodes[in_node].height
                           for in_node in node.in_nodes]) + 1 if node.in_nodes else 0
        if old_height != node.height:
            self.max_height = max(self.max_height, node.height)
            for out_node in node.out_connections:
                if self.nodes[out_node].height >= node.height and node != self.nodes[out_node]:
                    self.modify_height(out_node)

    def initialize_network(self) -> None:
        self.network = {}
        for node_id in self.nodes:
            if self.nodes[node_id].height not in self.network:
                self.network[self.nodes[node_id].height] = [node_id]
            else:
                self.network[self.nodes[node_id].height].append(node_id)

    def add_connection(self, connection: Connection) -> None:
        self.connections.append(connection)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def _initialize_nodes(self) -> None:
        self.input_nodes = [i for i in range(self.input_size)]
        self.output_nodes = [i for i in range(
            self.input_size, self.input_size+self.output_size)]

        self.nodes = {node_id: Node(node_id, INPUT if node_id in self.input_nodes else OUTPUT)
                      for node_id in self.input_nodes+self.output_nodes}
        for connection in self.connections:
            if connection.in_node not in self.nodes:
                self.add_node(Node(connection.in_node, HIDDEN))
            if connection.out_node not in self.nodes:
                self.add_node(Node(connection.out_node, HIDDEN))

    def _initialize_node_connections(self) -> None:
        for connection in self.connections:
            if not connection.disabled:
                self.nodes[connection.in_node].add_connection(connection)
                if not connection.recurrent:
                    self.nodes[connection.out_node].add_in_node(
                        connection.in_node)

    def _initialize_heights(self) -> None:
        self.max_height = 1
        for in_node in self.input_nodes:
            for out_node in self.nodes[in_node].out_connections:
                self.modify_height(out_node)

    def _fire_node(self, node_id: int) -> None:
        for out_node in self.nodes[node_id].out_connections:
            val = self._activate(
                self.nodes[node_id].out_connections[out_node]*self.nodes[node_id].val)
            if out_node == node_id or self.nodes[out_node].fired:
                # recurrent connection
                self.nodes[out_node].store(val)
            else:
                self.nodes[out_node].transfer(val)
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
        self.maximum_fitness = 0.0
        self.generations_without_improvement = 0
        self.total_adjusted_fitness = 0.0
        self.new = True

    def queue(self, genome: Genome) -> None:
        self.next_genomes.append(genome)

    def add(self, genome: Genome) -> None:
        self.genomes.append(genome)

    def next_generation(self) -> None:
        self.new = False
        self.genomes = self.next_genomes
        self.next_genomes = []

    def population_size(self) -> int:
        return len(self.genomes)

    def calculate_adjusted_fitnesses(self) -> None:
        self.total_adjusted_fitness = 0
        for genome in self.genomes:
            genome.adjusted_fitness = genome.fitness/len(self.genomes)
            self.total_adjusted_fitness += genome.adjusted_fitness

    def eliminate_lowest_fitness(self) -> None:
        self.genomes = sorted(self.genomes, key=lambda g: g.fitness, reverse=True)[
            1:] if len(self.genomes) > 0 else []

    def pick_highest_fitness(self) -> Genome:
        return [genome for genome in sorted(self.genomes, key=lambda g: g.fitness)][-1]

    def get_representative(self) -> Genome:
        return self.genomes[0]

    def pick_random(self) -> Genome:
        if len(self.genomes) == 1:
            return self.genomes[0]
        r = random()
        index = 0
        while True:
            r -= (self.genomes[index].adjusted_fitness /
                  self.total_adjusted_fitness) if self.total_adjusted_fitness != 0 else 1
            if r <= 0:
                break
            index += 1
        return self.genomes[index]

    def pick_random_two(self) -> Genome:
        if len(self.genomes) == 1:
            return self.genomes[0], self.genomes[0]
        elif len(self.genomes) == 2:
            return self.genomes[0], self.genomes[1]

        genome1 = self.pick_random()
        new_total = sum(
            [genome.adjusted_fitness for genome in self.genomes if genome != genome1])
        r = random()
        index = 0
        while True:
            if self.genomes[index] == genome1:
                index += 1
            r -= (self.genomes[index].adjusted_fitness /
                  new_total) if new_total != 0 else 1
            if r <= 0:
                break
            index += 1

        return genome1, self.genomes[index]


class NEAT:
    def __init__(self, input_size: int, output_size: int, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.config = {
            'population_size': 150,
            'inherit_disabled_rate': 0.75,
            'add_connection_rate': 0.05,
            'add_node_rate': 0.03,
            'connection_mutation_rate': 0.8,
            'perturb_weight_rate': 0.9,
            'c1':  1.0,
            'c2':  1.0,
            'c3':  0.4,
            'recurrent_connection_rate': 0.05,
            'delta_threshold': 3.0,
            'minimum_champion_size': 5,
            'species_expire_generations': 15,
            'mutation_without_crossover_rate': 0.25,
            'interspecies_mating_rate': 0.001,
            'weight_clamp': False,
            'weight_clamp_sigma': 5.0}
        self.config.update(kwargs)

        self.innovation_number = 0
        self.generation_innovations = {}

        self.generation_fitness = 0
        self.population_size = 0
        self.fitnesses = []
        self.species = []
        self._initialize_population()

    def genomes(self) -> [Genome]:
        return [genome for species in self.species for genome in species.genomes]

    def next_generation(self) -> None:
        self.population_size = 0
        self._update_species()
        self._calculate_adjusted_fitnesses()
        self._queue_next_population()
        self._replace_old_generation()

    def _update_species(self):
        for species in reversed(self.species):
            champion = species.pick_highest_fitness()
            if champion.fitness > species.maximum_fitness:
                species.maximum_fitness = champion.fitness
                species.generations_without_improvement = 0
            else:
                species.generations_without_improvement += 1
            species.eliminate_lowest_fitness()
            if species.generations_without_improvement >= self.config['species_expire_generations'] or species.population_size() == 0:
                self.species.remove(species)
            elif species.population_size() > self.config['minimum_champion_size']:
                self.population_size += 1
                species.queue(species.pick_highest_fitness())

    def _calculate_adjusted_fitnesses(self):
        self.generation_fitness = 0.0
        for species in self.species:
            species.calculate_adjusted_fitnesses()
            self.generation_fitness += species.total_adjusted_fitness

    def _queue_next_population(self):
        if len(self.species) == 0:
            self._initialize_population()
        else:
            while self.population_size < self.config['population_size']:
                self.population_size += 1
                species = self._pick_species()
                if species.population_size() < 2 or random() < self.config['mutation_without_crossover_rate']:
                    new_genome = species.pick_random().copy()
                else:
                    if random() < self.config['interspecies_mating_rate']:
                        genome1 = species.pick_random()
                        genome2 = self._pick_species().pick_random()
                    else:
                        genome1, genome2 = species.pick_random_two()
                    new_genome = self._cross_over(genome1, genome2)

                self._mutate(new_genome)
                new_genome.initialize_network()
                self._speciate(new_genome)

    def _pick_species(self) -> Species:
        r = random()
        index = 0
        while True:
            if self.species[index].new:
                index += 1
            r -= (self.species[index].total_adjusted_fitness /
                  self.generation_fitness) if self.generation_fitness != 0 else 1
            if r <= 0:
                break
            index += 1
        return self.species[index]

    def _initialize_population(self) -> None:
        for _ in range(self.config['population_size']):
            genome = Genome(self.input_size, self.output_size,
                            self._initial_connection())
            self._add_random_connection(genome)
            genome.initialize_network()
            self._speciate(genome)
        self._replace_old_generation()

    def _initial_connection(self) -> [Connection]:
        return [self._create_connection(choice([i for i in range(self.input_size)]), choice([i for i in range(self.input_size, self.input_size+self.output_size)]), gauss(0.0, 1.0), False, False)]

    def _speciate(self, genome: Genome) -> None:
        for species in self.species:
            if species.genomes and self._calculate_compatibility_distance(genome, species.get_representative()) <= self.config['delta_threshold']:
                species.queue(genome)
                break
        else:
            new_species = Species()
            new_species.add(genome)
            new_species.queue(genome)
            self.species.append(new_species)

    def _replace_old_generation(self) -> None:
        self.generation_innovations = {}
        new_species = []
        for species in self.species:
            if len(species.next_genomes) > 0:
                species.next_generation()
                new_species.append(species)
        self.species = new_species

    def _cross_over(self, genome1: Genome, genome2: Genome) -> Genome:
        equally_fit = genome1.adjusted_fitness == genome2.adjusted_fitness
        fitter_parent, weaker_parent = (
            genome1, genome2) if genome1.adjusted_fitness > genome2.adjusted_fitness else (genome2, genome1)

        combined_innovations = [innovation for innovation in sorted(
            set(fitter_parent.innovations()+weaker_parent.innovations()))]
        fitter_parent_genes = [fitter_parent.get_connection(
            innovation) for innovation in combined_innovations]
        weaker_parent_genes = [weaker_parent.get_connection(
            innovation) for innovation in combined_innovations]

        new_genes = []
        if equally_fit:
            for fitter, weaker in zip(fitter_parent_genes, weaker_parent_genes):
                picked = fitter if random() < 0.5 else weaker
                if picked:
                    new_genes.append(self._inherit_connection(picked))
        else:
            for fitter, weaker in zip(fitter_parent_genes, weaker_parent_genes):
                if fitter and weaker:
                    picked = fitter if random() < 0.5 else weaker
                    if picked:
                        new_genes.append(self._inherit_connection(picked))
                elif fitter:
                    new_genes.append(self._inherit_connection(fitter))
        return Genome(self.input_size, self.output_size, new_genes)

    def _inherit_connection(self, connection: Connection) -> None:
        new_connection = connection.copy()
        new_connection.disabled = connection.disabled and random(
        ) < self.config['inherit_disabled_rate']
        return new_connection

    def _mutate(self, genome: Genome) -> None:
        if random() < self.config['add_connection_rate']:
            self._add_random_connection(genome)
        if genome.connections and random() < self.config['add_node_rate']:
            self._add_random_node(genome)
        for connection in genome.connections:
            if random() < self.config['connection_mutation_rate']:
                self._perturb_weight(connection)

    def _add_random_connection(self, genome: Genome) -> None:
        if random() < self.config['recurrent_connection_rate']:
            in_node = choice(
                [node_id for node_id in genome.nodes if genome.nodes[node_id].node_type != INPUT])
            out_node = choice(
                [node_id for node_id in genome.nodes if genome.nodes[node_id].height <= genome.nodes[in_node].height])
            recurrent = True
        else:
            in_node = choice(
                [node_id for node_id in genome.nodes if genome.nodes[node_id].node_type != OUTPUT])
            out_node = choice([node_id for node_id in genome.nodes if genome.nodes[node_id].node_type !=
                               INPUT and genome.nodes[node_id].height >= genome.nodes[in_node].height])
            recurrent = False

        weight = gauss(0.0, 1.0)
        disabled = False

        # see if it exists, if it does, enable, otherwise, create
        existing_connection = genome.find_connection(in_node, out_node)
        if existing_connection:
            existing_connection.disabled = False
        else:
            genome.add_connection(self._create_connection(
                in_node, out_node, weight, disabled, recurrent))
            genome.modify_height(out_node)

    def _add_random_node(self, genome: Genome) -> None:
        # check if not recurrent
        old_connection = choice(
            [connection for connection in genome.connections if not connection.recurrent])
        old_connection.disabled = True
        new_node = self._create_node(genome)
        new_node.height = genome.nodes[old_connection.out_node].height
        genome.modify_height(old_connection.out_node)
        genome.add_node(new_node)
        genome.add_connection(self._create_connection(
            old_connection.in_node, new_node.id, 1.0))
        genome.add_connection(self._create_connection(
            new_node.id, old_connection.out_node, old_connection.weight))

    def _perturb_weight(self, connection: Connection) -> None:
        if random() < self.config['perturb_weight_rate']:
            connection.weight += gauss(0.0, 1.0)
        else:
            connection.weight = gauss(0.0, 2.0)
        connection.weight = max(min(
            connection.weight, self.config['weight_clamp_sigma']), -self.config['weight_clamp_sigma'])

    def _calculate_compatibility_distance(self, genome1: Genome, genome2: Genome) -> float:
        g1 = genome1.innovations()
        g2 = genome2.innovations()
        e = 0
        d = 0
        for i in sorted(set(g1+g2)):
            if i > max(g1) or i > max(g2):
                e += 1
            elif i in g1 != i in g2:
                d += 1
        w = 0
        m = 0
        for connection1 in genome1.connections:
            connection2 = genome2.get_connection(connection1.innovation)
            if connection2:
                w += abs(connection1.weight - connection2.weight)
                m += 1
        w = w / m if m != 0 else 0
        n = max(len(g1), len(g2))
        n = 1 if n < 20 else n
        return self.config['c1'] * e / n + self.config['c2'] * d / n + self.config['c3'] * w

    def _create_connection(self, in_node: int, out_node: int, weight: float, disabled: bool = False, recurrent: bool = False) -> Connection:
        in_out_pair = (in_node, out_node)
        if in_out_pair in self.generation_innovations:
            innovation = self.generation_innovations[in_out_pair]
        else:
            self.innovation_number += 1
            self.generation_innovations[in_out_pair] = self.innovation_number
            innovation = self.innovation_number
        return Connection(innovation,
                          in_node, out_node, weight, disabled, recurrent)

    def _create_node(self, genome: Genome):
        return Node(max([node_id for node_id in genome.nodes]) + 1, HIDDEN)

import unittest
import neat
import random


class TestNEATAlgorithm(unittest.TestCase):
    def test_copy_connection(self) -> None:
        connection = self._create_connection(0, 1, 2)
        copy = connection.copy()
        self.assertEqual(copy.innovation, 0)
        self.assertEqual(copy.in_node, 1)
        self.assertEqual(copy.out_node, 2)

    def test_create_genome_network(self) -> None:
        connections = [self._create_connection(0, 0, 3),
                       self._create_connection(1, 1, 3),
                       self._create_connection(2, 2, 4)]
        genome = self._create_genome(3, 2, connections)
        genome.initialize_network()
        self.assertEqual(set(genome.network[0]), set([0, 1, 2]))
        self.assertEqual(set(genome.network[1]), set([3, 4]))

    def test_create_genome_heights(self) -> None:
        connections = [self._create_connection(0, 0, 1),
                       self._create_connection(1, 0, 2),
                       self._create_connection(2, 2, 1)]
        genome = self._create_genome(1, 1, connections)
        genome.initialize_network()
        self.assertEqual(set(genome.network[0]), set([0]))
        self.assertEqual(set(genome.network[1]), set([2]))
        self.assertEqual(set(genome.network[2]), set([1]))
        self.assertEqual(genome.nodes[0].height, 0)
        self.assertEqual(genome.nodes[1].height, 2)
        self.assertEqual(genome.nodes[2].height, 1)

    def test_create_species(self) -> None:
        genomes = [self._create_genome(1, 1, [], 3),
                   self._create_genome(1, 1, [], 5),
                   self._create_genome(1, 1, [], 7),
                   self._create_genome(1, 1, [], 4)]

        species = neat.Species()
        for genome in genomes:
            species.add(genome)
        species.calculate_adjusted_fitnesses()
        self.assertEqual(species.total_adjusted_fitness, 3/4+5/4+7/4+4/4)

    def test_create_species_highest_fitness(self) -> None:
        genomes = [self._create_genome(1, 1, [], 3),
                   self._create_genome(1, 1, [], 5),
                   self._create_genome(1, 1, [], 7),
                   self._create_genome(1, 1, [], 4)]

        species = neat.Species()
        for genome in genomes:
            species.add(genome)
        species.calculate_adjusted_fitnesses()
        self.assertEqual(species.pick_highest_fitness().adjusted_fitness, 7/4)

    def test_calculate_compatibility_distance(self) -> None:
        connections1 = [self._create_connection(1, 0, 3),
                        self._create_connection(2, 1, 3),
                        self._create_connection(3, 2, 3),
                        self._create_connection(4, 1, 4),
                        self._create_connection(5, 4, 3),
                        self._create_connection(8, 0, 4)]
        connections2 = [self._create_connection(1, 0, 3),
                        self._create_connection(2, 1, 3),
                        self._create_connection(3, 2, 3),
                        self._create_connection(4, 1, 4),
                        self._create_connection(5, 4, 3),
                        self._create_connection(8, 0, 4),
                        self._create_connection(6, 4, 5),
                        self._create_connection(7, 5, 3),
                        self._create_connection(9, 2, 4),
                        self._create_connection(10, 0, 5)]
        genome1 = self._create_genome(3, 1, connections1)
        genome2 = self._create_genome(3, 1, connections2)
        # print(neat.NEAT(3, 1)._calculate_compatibility_distance(genome1, genome2))

    def test_create_neat(self) -> None:
        n = self._create_neat()
        self.assertEqual(len(n.genomes()), 1000)

    def test_calculate_adjusted_fitnesses(self) -> None:
        n = self._create_neat()
        for g in n.genomes():
            g.fitness += random.random()
        n._calculate_adjusted_fitnesses()

    def test_update_species(self) -> None:
        n = self._create_neat()
        for g in n.genomes():
            g.fitness += random.random()
        n._calculate_adjusted_fitnesses()

    def test_next_generation(self) -> None:
        n = self._create_neat()
        for g in n.genomes():
            g.fitness += random.random()
        n.next_generation()
        self.assertEqual(len(n.genomes()), 1000)

    def _create_neat(self, input_size: int = 3, output_size: int = 1) -> neat.NEAT:
        return neat.NEAT(input_size, output_size)

    def _create_genome(self, input_size: int, output_size: int, connections: [neat.Connection], fitness: float = 0.0) -> neat.Genome:
        genome = neat.Genome(input_size, output_size, connections)
        genome.fitness = fitness
        return genome

    def _create_connection(self, innovation: int, in_node: int, out_node: int, weight: float = 0.5, disabled: bool = False, recurrent: bool = False) -> neat.Connection:
        return neat.Connection(innovation, in_node, out_node, weight, disabled, recurrent)


if __name__ == '__main__':
    unittest.main()

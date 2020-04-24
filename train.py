import neat
import game
import random
import json

SAVE_FILE = './save.json'
NUM_GENERATIONS = 10000
SAVE_BEST = 100


class NeatGame:
    def __init__(self, genome: neat.Genome):
        self.genome = genome
        self.g = game.Game()

    def move(self):
        outputs = {i: weight for i, weight in enumerate(
            self.genome.feedforward(self._get_inputs()))}

        for i, weight in sorted(outputs.items(), key=lambda x: x[1], reverse=True):
            if self.g.check_available(i):
                self.g.move(i)
                break

    def alive(self):
        return not self.g.game_over

    def assign_fitness(self):
        self.genome.fitness = self.g.score

    def fitness(self):
        return self.genome.fitness

    def print_board(self):
        self.g.print_board()

    def _get_inputs(self):
        return [i/32768 for i in self.g.flattened_board()]


if __name__ == '__main__':
    n = neat.NEAT(16, 4)
    i = 0
    recent_best_genes = []
    highest_highest_fitness = 0
    highest_generation_fitness = 0
    while i < NUM_GENERATIONS:
        neat_games = [NeatGame(genome) for genome in n.genomes()]
        highest_fitness = 0
        for g in neat_games:
            while g.alive():
                g.move()
            g.assign_fitness()
            highest_fitness = max(highest_fitness, g.fitness())
        generation_fitness = n.generation_fitness
        print(
            f'Generation: {i} HighFitness: {highest_fitness} GenerationFitness: {generation_fitness}')
        if highest_fitness > highest_highest_fitness:
            highest_highest_fitness = highest_fitness
            recent_best_genes.append({
                'generation': i,
                'best_genome': sorted(n.genomes(), key=lambda genome: genome.fitness)[-1].as_list(),
                'highest_fitness': highest_fitness,
                'generation_fitness': generation_fitness
            })
            if len(recent_best_genes) > SAVE_BEST:
                recent_best_genes.pop(0)
        i += 1
        n.next_generation()

    # test and save
    # last_games = []
    # for genome in n.genomes():
    #     g = NeatGame(genome)
    #     while g.alive():
    #         g.move()
    #     g.assign_fitness()
    #     last_games.append({
    #         'genome': g.genome.as_list(),
    #         'score': g.fitness()
    #     })

    # last_games.sort(key=lambda g: g['score'])

    with open(SAVE_FILE, 'w') as save_file:
        save_file.write(json.dumps(recent_best_genes))

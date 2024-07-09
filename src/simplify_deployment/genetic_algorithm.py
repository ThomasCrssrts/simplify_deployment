import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from simplify_deployment.organism import Organism

# Create logger
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

# Create handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Add handler to logger
logger.addHandler(handler)


def genetic_algorithm(
    path_config: Path,
    y: pd.DataFrame,
    X: pd.DataFrame,
    population_size: int = 200,
    n_generations: int = 50,
    n_untouched: int = 1,
    reproduction_chance_second_over_first: float = 0.85,
    number_of_deaths: int = 10,
    mutation_chance: float = 1 / 1000,
    chance_of_random_variable_to_be_in_organism: float = 1 / 5000,
    extra_organisms=Optional[list[Path]],
    path_best_genome=Optional[Path],
) -> Organism:
    # Create starting population and populate them with random genomes.
    population = [
        Organism(path_config=path_config) for i in range(population_size)
    ]
    for organism in population:
        organism.init_random_genome(
            chance_of_random_variable_to_be_in_organism,
        )
    # Add extras to population
    if not (extra_organisms is None):
        population_extra = [
            Organism.from_yaml(
                path_config=path_config,
                path_genome=x,
            )
            for x in extra_organisms
        ]
        population += population_extra
        population_size = len(population)
    # Do until we have had all generations
    for n_generation in range(n_generations):
        #  Sort all the organisms by fitness
        for organism in population:
            organism.calculate_fitness(
                y=y,
                X_minute=X,
            )
        population = sorted(population, reverse=True)
        if not (path_best_genome is None):
            population[0].to_yaml(path_best_genome)
        logger.info(f"\nGeneration {n_generation}:")
        logger.info(f"Best fitness: {population[0].fitness}")
        logger.info(
            "Best organism used {} variables".format(
                population[0].get_n_variables_used(),
            )
        )
        vars_used = population[0].get_variables_as_list_of_str()
        logger.info("The variables used were:")
        for text in vars_used:
            logger.info("\n")
            logger.info(text)

        # Next step is reproduction.
        # We let the number of untouched go directly to next gen
        # The other ones need to fight for their place
        chance_to_get_picked = np.array(
            [
                reproduction_chance_second_over_first**n
                for n in range(population_size)
            ]
        )
        chance_to_get_picked = chance_to_get_picked / np.sum(
            chance_to_get_picked
        )
        partners = np.random.choice(
            population, size=population_size, p=chance_to_get_picked
        )

        # Now we know which partner every organism wants.
        # Let's create a new population with that
        # Sadly a number of them die and get replaced by new organisms
        # And another number get replaced by the top organisms
        # from previous gen so we can never get worse fitness
        offspring = [
            Organism.reproduce(a, b)
            for a, b in zip(
                population,
                partners,
            )
        ]

        # Let's also mutate the offspring
        for child in offspring:
            child.mutate(mutation_chance=mutation_chance)

        # Keep only the first ones. Others die or get replaced.
        if extra_organisms is None:
            n_extra = 0
        else:
            n_extra = len(extra_organisms)

        offspring = offspring[
            : (population_size - n_untouched - number_of_deaths - n_extra)
        ]
        offspring = offspring + population[:n_untouched]
        new_life = [Organism(path_config) for i in range(number_of_deaths)]
        for organism in new_life:
            organism.init_random_genome(
                chance_of_random_variable_to_be_in_organism,
            )
        # add extras again
        if not (extra_organisms is None):
            population_extra = [
                Organism.from_yaml(
                    path_config=path_config,
                    path_genome=x,
                )
                for x in extra_organisms
            ]
            offspring = offspring + new_life + population_extra
        else:
            offspring = offspring + new_life
        # Now offspring is the new generation and the circle of life continues
        population = offspring

    logger.info("\n All generations done. Retrieving the best organism.")
    for organism in population:
        organism.calculate_fitness(
            y=y,
            X_minute=X,
        )
    population = sorted(population, reverse=True)
    logger.info(
        "The final organism used {} variables".format(
            population[0].get_n_variables_used(),
        )
    )
    logger.info("The final best organism used following variables:")
    vars_used = population[0].get_variables_as_list_of_str()
    for text in vars_used:
        logger.info("\n")
        logger.info(text)

    return population[0]

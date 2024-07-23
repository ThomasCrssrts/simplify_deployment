from pathlib import Path

from simplify_deployment.data_wrangling import create_target, create_X
from simplify_deployment.genetic_algorithm import genetic_algorithm
from simplify_deployment.organism import Organism

path_config = Path(
    "src/simplify_deployment/config/lag_25.yaml",
)
X_minute = create_X(
    Path(
        "data/minute_train.parquet",
    ),
    Path(
        "data/qh_train.parquet",
    ),
)
y_minute = create_target(
    X_minute,
)

org = Organism(path_config)
org._init_empty_genome()
n_vars = org.get_n_variables_possible()
ga = genetic_algorithm(
    path_config=path_config,
    chance_of_random_variable_to_be_in_organism=1 / n_vars,
    mutation_chance=1 / n_vars,
    n_generations=50,
    n_untouched=1,
    number_of_deaths=10,
    population_size=200,
    reproduction_chance_second_over_first=0.85,
    X=X_minute,
    y=y_minute,
    extra_organisms=[
        Path(
            "src/simplify_deployment/genomes/lag_25_gen_1.yaml",
        ),
    ],
    path_best_genome=Path(
        "src/simplify_deployment/genomes/lag_25_best_genome_only_si.yaml",
    ),
)

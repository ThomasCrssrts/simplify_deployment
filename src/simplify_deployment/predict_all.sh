python src/simplify_deployment/predict_all.py \
    --path-config="/home/thomas/repos/simplify_deployment/src/simplify_deployment/config/lag_25.yaml" \
    --path-minute-data="/home/thomas/repos/simplify_deployment/data/simplify_1_0/minute_data.parquet" \
    --path-qh-data="/home/thomas/repos/simplify_deployment/data/simplify_1_0/quarter_data.parquet" \
    --path-best-genome="/home/thomas/repos/simplify_deployment/src/simplify_deployment/genomes/lag_25_best_genome.yaml" \
    --path-to-save-predictions="/home/thomas/repos/simplify_deployment/data/predictions.parquet" \
    --extra-organisms="/home/thomas/repos/simplify_deployment/src/simplify_deployment/genomes/lag_25_simplify_1_0.yaml" \
    --extra-organisms="/home/thomas/repos/simplify_deployment/src/simplify_deployment/genomes/lag_25_gen_2.yaml"



  
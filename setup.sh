conda create --name mcmc_env --file conda-linux-64.lock
eval "$(conda shell.bash hook)"
conda activate mcmc_env
poetry install
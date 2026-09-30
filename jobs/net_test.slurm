#!/bin/bash
# Checks whether a Grace compute node can reach the dataset hosts.
# Downloads nothing; only asks each host for its response headers.
#SBATCH --job-name=oxy_net_test
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=logs/%x_%j.out   # relative to the submit dir (repo root)

echo "Node: $(hostname)  Date: $(date)"
for url in https://www.kaggle.com https://drive.google.com https://github.com https://pypi.org/simple/; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$url")
    echo "$url -> HTTP ${code:-none}"
done

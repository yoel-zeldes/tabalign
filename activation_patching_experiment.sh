#!/bin/bash
datasets=("breast_cancer" "wine" "iris" "digits" "segment" "mfeat-factors" "vehicle" "ilpd" "credit-g" "sa-heart")
for ds in "${datasets[@]}"; do
    echo "--------------------------------------------------"
    echo "Running experiment for $ds..."
    echo "--------------------------------------------------"
    ./venv/bin/python3 activation_patching_experiment.py --dataset "$ds" --student_n 5 10 20 50 100 200 500
done

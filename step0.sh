#!/bin/bash

# Run this script first to generate most of the data

exit

# Generate data for figures 1 and 2
for i in 10 20 30 40 50 60
do
    python3 change_kurtosis_only.py newton $i
    python3 cond_vs_kurtosis.py newton $i
done

# Generate BoW data for figures 3 and 5
for fname in enron kos nips nytimes
do
    python3 doc_dumps.py newton $fname
    python3 doc_dumps.py todd $fname
done

# Generate MNIST data for figures 3 and 5
python mnist.py newton
python mnist.py todd

# Generate remaining data for figure 5
python3 change_kurtosis_only.py todd 50
python3 cond_vs_kurtosis.py todd 50

# Generate data related to hybrid methods
python3 perfect_hybrid.py
python3 hybrid_create_metadata.py

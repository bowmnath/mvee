#!/bin/bash

# Plot figure 5
# Data must already be generated
python3 thesis_generate_kurtosis.py newton 50 -r -i
python3 thesis_generate_kurtosis.py newton 50 -r -v -i
python3 thesis_generate_kurtosis.py todd 50 -r -i
python3 thesis_generate_kurtosis.py todd 50 -r -v -i

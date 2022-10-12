#!/bin/bash

# Plot figure 1
# Data must already be generated
python3 thesis_generate_kurtosis.py newton 10 -u
python3 thesis_generate_kurtosis.py newton 10 -u -v

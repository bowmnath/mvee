#!/bin/bash

# Plot figure 3
# Data must already be generated
python3 thesis_generate_kurtosis.py newton 50 -r
python3 thesis_generate_kurtosis.py newton 50 -r -v

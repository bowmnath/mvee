#!/bin/bash

# Generate data and plot figure 4
# Runs for about a minute, then displays first plot
# After you close first plot, runs for another minute,
# then displays second plot
python3 change_kurtosis_n.py newton
python3 change_kurtosis_n.py todd

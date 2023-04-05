#!/bin/bash

for kurtosis in low very high
do
    python3 scaling_m_n.py $kurtosis m 10 3 6 8 100
    python3 scaling_m_n.py $kurtosis n 10 1 2 10 1000000
done

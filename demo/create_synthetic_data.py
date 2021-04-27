# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:38:24 2021

@author: Federico Calesella
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

filepath = 'path/file_name.xlsx'

x, y = make_classification(n_samples=100, n_features=20, n_informative=2, 
                           n_redundant=2, n_repeated=0, n_classes=2, 
                           n_clusters_per_class=2, weights=None, flip_y=0.01, 
                           class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, 
                           shuffle=True, random_state=None)
data = np.column_stack([y, x])
df = pd.DataFrame(data)

df.to_excel(filepath, index=False)
# -*- coding: utf-8 -*-
"""
@author: Federico Calesella

"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

filepath = 'path/file_name.xlsx'

x, y = make_regression(n_samples=100, n_features=20, n_informative=2, 
                       n_targets=1, bias=0.0, effective_rank=None, 
                       tail_strength=0.5, noise=0.0, shuffle=True, coef=False, 
                       random_state=None)
data = np.column_stack([y, x])
df = pd.DataFrame(data)

df.to_excel(filepath, index=False)

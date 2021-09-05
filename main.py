#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:22:28 2021

@author: duboisantoine
"""
import numpy as np
from RLdesign import LearningDesign



if __name__ == "__main__":
    J = 10
    p = 4
    X = np.random.standard_t(2, size = (J, p))
    mixed_effect = X[:,0] * X[:,1] # introduction of mixed effects
    X = np.column_stack([X, mixed_effect])
    
    cov = np.random.rand(J, J)
    cov = np.dot(cov, cov.T)
    
    choice_set_size = 3
    nb_observations = 50
    

    learning = LearningDesign(X, nb_observations, choice_set_size, accuracy=100, B=10)
    optimal_design, optimal_score = learning.train(max_episodes = 50, criterion="D")
    learning.summary(log=True)
        

            




















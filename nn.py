#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:56:37 2021

@author: duboisantoine
"""
import logging 
logging.getLogger('tensorflow').disabled = True # disable the tensorflow warning error
import tensorflow as tf
import math


class BDQ(tf.keras.Model):
    def __init__(self, nb_observations, choice_set_size, nb_alternatives):
        super(BDQ, self).__init__()
        self._N = nb_observations
        self._csz = choice_set_size
        self._J = nb_alternatives
        
        # flatten layer
        self.flatten = tf.keras.layers.Flatten()
        
        # state representation layers
        self.state_representation_l1 = tf.keras.layers.Dense( self._N, activation="relu", kernel_initializer="HeNormal")
        self.state_representation_l2 = tf.keras.layers.Dense( self._N //2, activation="relu", kernel_initializer="HeNormal")
        
        # state valuation layers
        self.state_valuation_l = tf.keras.layers.Dense( self._N //2, activation="relu", kernel_initializer="HeNormal")
    
        # advantages layers
        self.advantage_l = tf.keras.layers.Dense( self._N //2, activation="relu", kernel_initializer="HeNormal")
        
        # Q-values layers
        self.qvalue_l = tf.keras.layers.Dense( self._J, activation="softmax", kernel_initializer="HeNormal")
        
        # utils
        self.concat = tf.keras.layers.Concatenate()
        
    def call(self, states):
        # flatten at init
        states = self.flatten(states)
        
        # state representation
        state_rep = self.state_representation_l1(states)
        state_rep = self.state_representation_l2(state_rep)
        
        # state valuation
        state_value = self.state_valuation_l(state_rep)
        
        # advantages and Q-values
        actions = []
        for i in range(self._csz):
            action = self.advantage_l(state_rep)
            action = self.concat([state_value, action])
            action = self.qvalue_l(action)
            actions.append(action)            
        
        # output
        out = self.concat(actions) 
        return out
    
    def model(self):
        x = tf.keras.layers.Input(shape=(self._N, self._csz))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    



class DNN(tf.keras.Model):
    def __init__(self, nb_observations, choice_set_size, nb_alternatives):
        super(DNN, self).__init__()
        self._N = nb_observations

        self._combi = math.comb(nb_alternatives, choice_set_size)

        self.dense_l1 = tf.keras.layers.Dense( self._N//2, activation="relu", kernel_initializer="HeNormal")
        self.dense_l2 = tf.keras.layers.Dense(self._combi, activation="softmax", kernel_initializer="HeNormal")
        
    def call(self, states):
        x = self.dense_l1(states)
        x = self.dense_l2(x)
        return x
    
    def model(self):
        x = tf.keras.layers.Input(shape=(self._N, ))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))





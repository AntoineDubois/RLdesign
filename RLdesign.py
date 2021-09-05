#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 16:22:00 2021

@author: duboisantoine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:47:17 2021

@author: duboisantoine
"""

# -*- coding: utf-8 -*-
from collections import deque
from itertools import combinations
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import warnings

from nn import DNN

    



class LearningDesign:
    def __init__(self, X, nb_observations, choice_set_size, accuracy, B):
        self._N = nb_observations
        self._csz = choice_set_size
        self.X = X
        self._J, self._p = X.shape
        self._acc = accuracy
        self._B = B
                
        
        self.action_set = np.asarray(list(combinations(list(range(0,self._J)), self._csz)))
        self._combi = self.action_set.shape[0]
        
        self.NN = DNN(self._N, self._csz, self._J).model()
        self.NN.load_weights("weights/modelDNN.h5")
        
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.indiv_info()
        


    def indiv_info(self):
        self.info_list = list()
        for index in self.action_set:
            M = 0
            theta = np.random.normal( loc=0, scale=1, size=( self.X.shape[1], self._acc))
            for i in range(self._acc):
                exp = np.exp( np.dot(self.X[index], theta[:, i]))
                sum_exp = np.sum(exp)
                proba = exp / sum_exp
                proba = np.asmatrix(proba)
                matrix_proba = np.dot(proba.T, proba)
                diag_proba = np.diag(np.asarray(proba).flatten())
                M += np.dot( self.X[index].T, np.dot(diag_proba -matrix_proba, self.X[index]))
            self.info_list.append(M/self._acc)



    def act(self, state):
        with tf.GradientTape() as tape:
            prob = self.NN(state, training=True) #I did know it could work like that
            prob = tf.clip_by_value( prob, 1e-37, 1)
            dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        delta_log_prob = tape.gradient(log_prob, self.NN.trainable_variables)
        #print("act", np.array([(tf.math.is_nan(delta_log_prob[idx]).numpy()).all() for idx in range(len(delta_log_prob))]))
        return int(action.numpy()[0]), delta_log_prob
    
    
    
    def gen_traj(self):
        sample_traj = {}
        sample_cost = list()
        
        for b in range(self._B):
            sample_traj[b] = tf.Variable(tf.zeros((1, self._N), dtype=tf.int32))
            
            #init
            new_action, delta_log_prob = self.act(sample_traj[b])
            sum_grad_proba_log = delta_log_prob
            sample_traj[b][0, 0].assign(new_action + 1) # +1 because 0 means no action
            info_matrix = self.info_list[new_action]
            
            #heredity
            for i in range(1, self._N):
                new_action, delta_log_prob = self.act(sample_traj[b])
                sum_grad_proba_log = [sum_grad_proba_log[idx] + delta_log_prob[idx] for idx in range(len(delta_log_prob))]
                sample_traj[b][0, i].assign(new_action + 1) # +1 because 0 means no action
                info_matrix += self.info_list[new_action]
            
            cost = self.cost(info_matrix)
                        
            experience = [sum_grad_proba_log[idx] * (cost -self.base_line[b]) for idx in range(len(sum_grad_proba_log))]
            
            sample_cost.append(cost) 
            if b == 0:
                sample_av_experience = experience
            else:
                sample_av_experience = [(sample_av_experience[idx] + experience[idx])/self._B for idx in range(len(experience))]
            
        return sample_traj, sample_cost, sample_av_experience
    
                
    def train(self, max_episodes=100, plot=False, base_line=True, criterion="D", convergence=10, slow=True):
        if not criterion in ["D", "E", "T"]:
            raise Exception("unknown criterion")
        elif criterion == "D":
            if slow:
                def cost(info_matrix):
                    return np.exp(- np.log(max(0, np.linalg.det(info_matrix))))
                
            else:
                def cost(info_matrix):
                    return np.exp(- max(0, np.linalg.det(info_matrix)))
            
            def score(cost):
                if cost == 0:
                    return float("inf")
                else:
                    return 1/cost
            self.cost = cost
            self.score = score
            self.criterion = "D"
            
        elif criterion == "E":
            if slow:
                def cost(info_matrix):
                    return np.exp(- np.log(max(0, min(np.linalg.eigvals(info_matrix)))))
            else:
                def cost(info_matrix):
                    return np.exp(- max(0, min(np.linalg.eigvals(info_matrix))))
            
            def score(cost):
                if cost == 0:
                    return float("inf")
                else:
                    return 1/cost
                
            self.cost = cost
            self.score = score
            self.criterion = "E"
            
        elif criterion == "T":
            def cost(info_matrix):
                return 1/np.trace(info_matrix)
            
            def score(cost):
                if cost == 0:
                    return float("inf")
                else:
                    return 1/cost
                
            self.cost = cost
            self.score = score
            self.criterion = "T"
           
        
        traj_ref = tf.zeros((1,self._N), dtype=tf.int32)
        cost_ref = float("inf")
        alpha = 0.05
        self.history_score_ref = deque([0]) # 0=score_ref
        if base_line:
            self.base_line = np.random.rand(self._B)
        else:
            self.base_line = np.zeros(self._B)
        

        no_improvement = 0
        episode = 0
        self.score_is_finite = True
        while( self.score_is_finite and no_improvement<convergence and episode < max_episodes ):
            no_improvement += 1
            episode += 1
            
            sample_traj, sample_cost, sample_av_experience = self.gen_traj()

            is_nan = np.array([(tf.math.is_nan(sample_av_experience[idx]).numpy()).all() for idx in range(len(sample_av_experience))])
            if (is_nan != False).any():
                raise Exception("nan values in the policy gradient")
            
            best_traj = np.argmin(sample_cost)
            if sample_cost[best_traj] < cost_ref:
                traj_ref = sample_traj[best_traj]
                cost_ref = sample_cost[best_traj]
                no_improvement = 0
            
            score = self.score(cost_ref)
            self.history_score_ref.append(score)
            if np.isinf(score): # introduction of some convergence criterion 
                self.score_is_finite = False
                             
            self.optimizer.apply_gradients(zip(sample_av_experience, self.NN.trainable_variables))
            self.base_line = alpha* self.base_line + (1 -alpha)*np.mean(self.base_line)
            
            
        if not episode < max_episodes:
            warnings.warn("The maximum number of episodes is reached")
        if not no_improvement<convergence:
            warnings.warn("The maximum number of episodes without improvement is reached")
        if not self.score_is_finite: # to reactivate
            print("The score is infinite after", episode, "episodes")
        
        
        optimal_design = self.action_set[traj_ref.numpy().flatten()-1]
        self._optimal_score = self.score(cost_ref)
        if plot:
            self.summary(log=True)
        
        return optimal_design, self._optimal_score #return the best combination after the episodes
        
    def summary(self, log=False):
        if (np.isinf(self.history_score_ref)).any():
            self.history_score_ref.pop()
            if log:
                plt.plot(list(range(len(self.history_score_ref))), np.log(self.history_score_ref))
                plt.xlabel("episodes")
                plt.ylabel("log-" + self.criterion + "-score")
                plt.show()
            else:
                plt.plot(list(range(len( self.history_score_ref ))), self.history_score_ref)
                plt.xlabel("episodes")
                plt.ylabel(self.criterion + "-score")
                plt.show()
            print("Optimal " + self.criterion + " -score:", self._optimal_score)
        else:
            if log:
                plt.plot(list(range(len(self.history_score_ref))), np.log(self.history_score_ref))
                plt.xlabel("episodes")
                plt.ylabel("log-" + self.criterion + "-score")
                plt.show()
            else:
                plt.plot(list(range(len(self.history_score_ref))), self.history_score_ref)
                plt.xlabel("episodes")
                plt.ylabel(self.criterion + "-score")
                plt.show()
            print("Optimal" + self.criterion + " -score:", self._optimal_score)

    













    

"""
build a stacked generalizer/blender model using an arbitrary number of inputs
"""

from sklearn.model_selection import KFold
from copy import copy
import numpy as np

class Blender:
    def __init__(self, num_levels, learner_dict, verbose=False):
        self.num_levels = num_levels
        self.learner_dict = learner_dict
        self.verbose = verbose

    def fit(self, X, y, num_folds=None): 
        # this will be a list of lists
        # outer list is levels
        # inner lists are the learners for that level
        
        if not num_folds:
            num_folds = X.shape[0]
            
        self.final_model = []
            
        feature_space = X.copy()
        output_space = y.copy()
        
        for level in range(self.num_levels-1):
            this_levels_learners = self.learner_dict[level]
            trained_learners, feature_space, output_space = \
                                self.low_level_fit(self.learner_dict[level], feature_space, output_space, num_folds)
            
            self.final_model.append(trained_learners)

        # get the final learner
        meta_learner = self.learner_dict[self.num_levels - 1]
        
        if self.verbose:
            print("Training meta-learner...", end="")
        
        meta_learner.fit(feature_space, output_space)
        
        if self.verbose:
            print("Done")

        self.final_model.append(meta_learner)
            
    def low_level_fit(self, learners, X, y, num_folds):
        # takes a list of learners. fits them all it to k folds. 
        # returns the new feature space, and the trained models.
        
        splitter = KFold(n_splits = num_folds)
        
        trained_learners = []
        new_feature_space = []
        new_output_space = []
        
        for learner_num, learner in enumerate(learners, 1):
            # intantiate list of outputs for this learner
            learner_feature_space = []
            learner_output_prob_zero = []
            learner_output_prob_one = []
            
            if self.verbose:
                print("Training low level learner {}...".format(learner_num))
            
            for fold_num, idx in enumerate(splitter.split(X), 1):
                if self.verbose:
                    print("\tTraining fold number {} out of {}...".format(fold_num, num_folds), end="")
                
                X_tr, X_te, y_tr, y_te = X[idx[0]], X[idx[1]], y[idx[0]], y[idx[1]]

                learner.fit(X_tr, y_tr)
                
                y_output = learner.predict_proba(X_te)
                
                if len(learner_feature_space) == 0:
                    learner_feature_space = y_output.copy()
                    
                else:
                    learner_feature_space = np.concatenate((learner_feature_space, y_output), axis = 0)

                '''
                learner_output_prob_zero.append(y_output[0][0])
                learner_output_prob_one.append(y_output[0][1])
                '''
                
                if self.verbose:
                    print("Done")
                
                if learner_num == 1:
                    if len(new_output_space) == 0:
                        new_output_space = y_te.copy()
                        
                    else:
                        new_output_space = np.concatenate((new_output_space, y_te), axis=0)
                    
            if len(new_feature_space) == 0:
                new_feature_space = learner_feature_space.copy()
                
            else:
                new_feature_space = np.concatenate((new_feature_space, learner_feature_space), axis=1)
                
            if self.verbose:
                print("Done.\n")
                
            learner.fit(X, y)
            
            trained_learners.append(copy(learner))
        
        return trained_learners, new_feature_space, new_output_space
            
    def predict(self, X):
        X_prev = X.copy()

        for level, learner_list in enumerate(self.final_model):
            if level < self.num_levels - 1:
                X_update = np.zeros((X.shape[0], len(learner_list)*2))
                
                shift = 0

                for learner_num, learner in enumerate(learner_list):
                    prediction = learner.predict_proba(X_prev)
                
                    X_update[:,learner_num+shift] = prediction[:,0]
                    X_update[:,learner_num+shift+1] = prediction[:,1]
                    
                    shift+=1
                    
                X_prev = X_update.copy()
                    
            else:
                return self.final_model[level].predict(X_update)
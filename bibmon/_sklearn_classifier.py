# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:40:00 2025

@author: leovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report 

from ._generic_model import GenericModel

###############################################################################

class sklearnClassifier (GenericModel):
    """
    Interface for sklearn classifiers.
            
    Parameters
    ----------
    classifier: any classifier that uses the sklearn interface. 
        For example:
            * sklearn.ensemble.forest.RandomForestClassifier,
            * sklearn.neural_network.multilayer_perceptron.MLPClassifier,
            * etc....
    permutation_importance: boolean, optional
        Whether permutation variable importance should be calculated.    
        """     

    ###########################################################################

    def __init__ (self, classifier, permutation_importance = False):

        self.has_Y = True       
        self.classifier = classifier
	
        self.name = self.classifier.__class__.__name__
        
        self.permutation_importance = permutation_importance
    
    ###########################################################################
        
    def train_core (self):
        
        self.classifier.fit(self.X_train.values,
                           self.Y_train.values.squeeze())
        
        if self.permutation_importance:
            
            res = permutation_importance(self.classifier, 
                                         self.X_train.values, 
                                         self.Y_train.values.squeeze(),
                                         n_repeats=10)
            
            self.classifier.perm_feature_importances_ = res.importances_mean

    ###########################################################################

    def map_from_X (self, X):
        
        return self.classifier.predict(X)
    
    ###########################################################################
    
    def set_hyperparameters (self, params_dict):   
        
        for key, value in params_dict.items():
            setattr(self.classifier, key, value)
            
    ###########################################################################
            
    def update_importances(self):
        """
        Calculates permutation importances of the variables.
        """          

        res = permutation_importance(self.classifier, 
                                     self.X_test.values, 
                                     self.Y_test.values.squeeze(),
                                     n_repeats = 10)
            
        self.classifier.perm_feature_importances_ = res.importances_mean        
            
    ###########################################################################

    def plot_importances(self, n = None, permutation_importance = False):
        
        """
        Plots the permutation importances of the variables.
    
        Parameters
        ----------
        n: int, optional
            Maximum number of variables to be plotted.
        permutation_importance: boolean, optional
            If permutation importances should be prioritized over coefficients
            in linear models.
        """          
        
        model = self.classifier
        
        if hasattr(model,'coef_'):
            imp = model.coef_
        elif hasattr(model,'feature_importances_'):
            imp = model.feature_importances_
        elif (hasattr(model,'perm_feature_importances_')):
            imp = model.perm_feature_importances_

        if ((hasattr(model,'coef_') or
            hasattr(model,'feature_importances_')) and permutation_importance): 
            if hasattr(model, 'perm_feature_importances_'):
                imp = model.perm_feature_importances_
          
        if not (hasattr(model,'coef_') or
                hasattr(model,'feature_importances_') or
                hasattr(model, 'perm_feature_importances_')):
            print('There are no importances calculated for this model.')
            return
        
        tags = self.X_train.columns
        
        if n is not None:
            pass
        else:
            n = len(self.X_train.columns)
    
        fig, ax = plt.subplots(1,2, figsize = (20,4))
    
        coefs = []
        abs_coefs = []
    
        coefs = (pd.Series(imp, index = tags))
        coefs.plot(use_index=False, ax=ax[0]);
        abs_coefs = (abs(coefs)/(abs(coefs).sum()))
        abs_coefs.sort_values(ascending=False).plot(use_index=False, ax=ax[1],
                                                    marker='.')
    
        ax[0].set_title('Relative variable importances')
        ax[1].set_title('Relative variable importances - \
                        descending order')
    
        abs_coefs_df = pd.DataFrame(np.array(abs_coefs).T,
                                    columns = ['Importances'],
                                    index = tags)
    
        df = abs_coefs_df['Importances'].sort_values(ascending=False)
        
        plt.figure()
        df.iloc[0:n].plot(kind='barh', figsize=(15,0.25*n), legend=False)
        
        return df
    
    ###########################################################################
    def class_metrics_report(self):
        """
        Generates the overall metrics report of the classification task
        
        This function takes the base model classifier
        and compares the testing data with predictions to 
        generate the reporting metrics.
    
        Parameters
        ----------
        
        """
        
        #Importing the model classifier instance
        model=self.classifier
        y_pred=model.predict(self.X_test.values)
        y_test=self.Y_test.values.squeeze()
            
        report=classification_report(y_test, y_pred, target_names=y_test.columns)
        
        return report
    
    def system_deterioration(self,window_size=None):
        
        """
        Provides a chart for system deterioraration based on
        the fault probability of the classifier model
    
        Parameters
        ----------
        
        window_size: int, optional
        
        defining a windows size rolling for the time series data
        to compute the window average
        
        """
        
        model=self.classifier
        
        prob_fault=model.predict_proba(self.X_test.values)[:, 1]
        
        # Checking if the index is a timestamp
        if not pd.api.types.is_datetime64_any_dtype(self.X_test.index):
            print("Warning: Index is not a timestamp. \
                  Creating a dummy datetime index.")
            self.X_test.index = pd.date_range(start="2000-01-01", 
            periods=len(self.X_test), freq="H")
            
        fault_prob_series=pd.Series(prob_fault,index=self.X_test.index)
        
        # Apply a rolling window to smooth the data
        
        rolling_window=None
        if window_size is not None:
            rolling_window = fault_prob_series.rolling(window=window_size,
                                                   min_periods=1).mean()
            
        return rolling_window,fault_prob_series
        
    def plot_system_deterioration(self, ax=None, logy= False, legends= True,
                           plot_threshold= True, threshold= 0.5):
        
        """
        Plot the temporal evolution of fault probabilities 
        for system deterioration monitoring.

        ax: matplotlib.axes._subplots.AxesSubplot, optional
            Axis on which the graph will be plotted.
        logy: If True, use a logarithmic scale for the y-axis.
        legends: If True, display legends on the plot.
        plot_threshold: If True, plot the threshold line.
        threshold: The threshold value for fault probability.
        """
        
        # Get fault probabilities and rolling window
        fault_prob_series, rolling_window = self.system_deterioration()
        
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots()
            
        # Plot fault probabilities
        fault_prob_series.plot(ax=ax, logy=logy, ls='', marker='.', 
                               label='Fault Probability')
        
        # Plot rolling window if available
        if rolling_window is not None:
            rolling_window.plot(ax=ax, logy=logy, color='red', 
            label=f'Rolling Average ({self.window_size}h)')
    
        if plot_threshold:
            ax.axhline(y=threshold, color='black', ls='--', 
                       label=f'Threshold ({threshold})')
        
        if legends:
            ax.legends(fontsize=12)
            
        ax.set_xlabel('Time')
        ax.set_ylabel('Fault Probability')
        ax.set_title('System Deterioration Monitoring')
        
        
        
        
        
        
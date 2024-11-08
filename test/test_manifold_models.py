# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:34:01 2024

@author: Leonardo Voltolini
"""

import bibmon
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

SC=StandardScaler()

# loading the data from TEP
df_train, df_test = bibmon.load_tennessee_eastman(train_id = 0, 
                                                   test_id = 1)

df=pd.concat([df_train,df_test])

# preprocessing pipeline    
f_pp = ['normalize']

#X_train=bibmon.PreProcess.normalize(df=train_df)

X_train=SC.fit_transform(df_train)

X_test=SC.transform(df_test)

X=np.concatenate( (X_train, X_test),axis=0)

 
for attr in bibmon.__all__:             
    a = getattr(bibmon,attr)     
    if isinstance(a, type):         
        if a.__base__ == bibmon._generic_model.GenericModel:   
            if a == bibmon.sklearnManifold:                 
                from sklearn.manifold import TSNE
                m = a(TSNE(n_components=2))
            else:                    
                m = a()        
                          
            # TRAINING
                
            X_embedded=m.fit_transform(df, f_pp=f_pp)
            X_embedded.plot_embedding()
            
            
                
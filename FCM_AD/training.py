import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split 
from Outlier_Function import main_func, data_managing,best_cluster,euclidean_distance,ada_bost_weights

# from Adaboost_Function import adaboost_weights


"***************** Dataset****************************"  
random.seed(10)          
iris = pd.read_csv("iris.csv" , sep=',')
columns = list(iris.columns) 
features = columns[:len(columns) - 1] 
x_features = iris.drop(['target'],axis=1)  
y_labels = iris['target']

"***************** Data Split****************************" 
clus_num = 3
x_train,x_test,y_train,y_test = train_test_split(x_features,y_labels, train_size=0.8, test_size=0.2, random_state=0)

"***************** Fuzzy_clustering ****************************"  
train_labels, train_centers, train_data, train_membership = main_func(x_train, clus_num) 
updated_train = data_managing(x_train, train_membership) 
np_train_centers = np.asarray(train_centers)
no_colum = len(iris.columns)
dist_list = []
for item in updated_train:
    clus_no = str(item[-2:-1]) 
    clus_no = int(clus_no[1]) 
    pt1 = item[:-2]           
    center_pt = np_train_centers[clus_no]   
    # print(pt1, center_pt)
    dist = euclidean_distance(pt1,center_pt) 
    # print(dist)
    dist_list.append(dist)
        
updated_train = pd.DataFrame(updated_train) 
dist_list = pd.DataFrame(dist_list)

updated_train = updated_train.to_numpy() 
dist_list = dist_list.to_numpy()

comp_data = np.concatenate((updated_train, dist_list), axis=1) 

results = pd.DataFrame(comp_data) 
results.to_excel('Results_file.xlsx', index = False)
xxx = results.to_numpy()

# dist_list = []
# for item in xxx:
#     distance_val = np.squeeze(item[-1]) #select clus no from 2nd last
#     # print(distance_val)
#     if distance_val > 1.0:
#         pt1 = item[:-2]
#         print(pt1)
#         dist_list.append(pt1)
    
    # rend_state = distance_val>1

"***************** Adaboosted_weights ****************************"         

xx_train,xx_test,_,_ = train_test_split(comp_data,comp_data, train_size=0.8, test_size=0.2, random_state=0)

train_feature = xx_train[:,:no_colum-1]
train_target = xx_train[:,no_colum-1]
train_weights_x = xx_train[:,no_colum]

test_feature = xx_test[:,:no_colum-1]
test_target = xx_test[:,no_colum-1]
test_weights = xx_test[:,no_colum]

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((train_feature, train_target,train_weights_x))
train_dataset = train_dataset.batch(batch_size)


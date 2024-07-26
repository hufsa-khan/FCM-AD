
import math
import copy
import random
import operator
import numpy as np
from numpy import *
import pandas as pd
from fcmeans import FCM
from sklearn import tree
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


print("===================Adaboost function to get the weights =====================")

def ada_bost_weights(dataset):
    no_rows = len(dataset)
    no_colum = len(dataset.columns)
      
    dataset['probR1'] = 1/(dataset.shape[0])
    
    #shuffling the dataset
    random.seed(10)
    dataset_1 = dataset.sample(len(dataset), replace = True, weights = dataset['probR1'])
    
    #X_train and Y_train split
    X_train = dataset_1.iloc[0:no_rows,0:no_colum-1]
    y_train = dataset_1.iloc[0:no_rows,no_colum-1]
    
    #fitting the DT model with depth one
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100)
    clf = clf_gini.fit(X_train, y_train)
    
    y_pred = clf_gini.predict(dataset.iloc[0:no_rows,0:no_colum-1])
    dataset['pred1'] = y_pred
    
    #misclassified = 0 if the label and prediction are same
    dataset.loc[dataset.target != dataset.pred1, 'misclassified'] = 1
    dataset.loc[dataset.target == dataset.pred1, 'misclassified'] = 0
    
    e1 = sum(dataset['misclassified'] * dataset['probR1'])
    alpha1 = 0.5*log((1-e1)/e1)
    
    #update weight
    new_weight = dataset['probR1']*np.exp(-1*alpha1*dataset['target']*dataset['pred1'])
    
    #normalized weight
    z = sum(new_weight)
    normalized_weight = new_weight/sum(new_weight)
    dataset['prob2'] = round(normalized_weight,4)
    
    #round 2
    random.seed(70)
    #shuffle the dataset
    dataset2 = dataset.sample(len(dataset), replace = True, weights = dataset['prob2'])
    
    dataset2 = dataset2.iloc[:,0:no_colum]
    X_train = dataset2.iloc[0:no_rows,0:no_colum-1]
    y_train = dataset2.iloc[0:no_rows,no_colum-1]
    
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 7097)
    clf = clf_gini.fit(X_train, y_train)
    y_pred = clf_gini.predict(dataset2.iloc[0:no_rows,0:no_colum-1])
    #adding a column pred2 after the second round of boosting
    dataset['pred2'] = y_pred
    
    #adding a field misclassified2
    dataset.loc[dataset.target != dataset.pred2, 'misclassified2'] = 1
    dataset.loc[dataset.target == dataset.pred2, 'misclassified2'] = 0
    
    e2 = sum(dataset['misclassified2'] * dataset['prob2'])
    alpha2 = 0.5*log((1-e2)/e2)
    
    #update weight
    new_weight = dataset['prob2']*np.exp(-1*alpha2*dataset['target']*dataset['pred2'])
    z = sum(new_weight)
    normalized_weight = new_weight/sum(new_weight)
    dataset['prob3'] = round(normalized_weight,4)
    
    #round 3
    random.seed(30)
    dataset3 = dataset.sample(len(dataset), replace = True, weights = dataset['prob3'])
    dataset3 = dataset3.iloc[:,0:no_colum]
    X_train = dataset3.iloc[0:no_rows,0:no_colum-1]
    y_train = dataset3.iloc[0:no_rows,no_colum-1]
    
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 89076)
    clf = clf_gini.fit(X_train, y_train)
    
    #adding a column pred3 after the third round of boosting
    y_pred = clf_gini.predict(dataset.iloc[0:no_rows,0:no_colum-1])
    dataset['pred3'] = y_pred
    
    #adding a field misclassified3
    dataset.loc[dataset.target != dataset.pred3, 'misclassified3'] = 1
    dataset.loc[dataset.target == dataset.pred3, 'misclassified3'] = 0
    
    #weighted error calculation
    e3 = sum(dataset['misclassified3'] * dataset['prob3'])
    alpha3 = 0.5*log((1-e3)/e3)
    
    #update weight
    new_weight = dataset['prob3']*np.exp(-1*alpha3*dataset['target']*dataset['pred3'])
    z = sum(new_weight)
    normalized_weight = new_weight/sum(new_weight)
    dataset['prob4'] = round(normalized_weight,4)
    
    #Round 4
    random.seed(40)
    dataset4 = dataset.sample(len(dataset), replace = True, weights = dataset['prob4'])
    dataset4 = dataset4.iloc[:,0:no_colum]
    X_train = dataset4.iloc[0:no_rows,0:no_colum-1]
    y_train = dataset4.iloc[0:no_rows,no_colum-1]
    
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 6723)
    clf = clf_gini.fit(X_train, y_train)
    
    #adding a column pred4 after the fourth round of boosting
    y_pred = clf_gini.predict(dataset.iloc[0:no_rows,0:no_colum-1])
    dataset['pred4'] = y_pred
    
    #adding a field misclassified4
    dataset.loc[dataset.target != dataset.pred4, 'misclassified4'] = 1
    dataset.loc[dataset.target == dataset.pred4, 'misclassified4'] = 0
    
    #error calculation
    e4 = sum(dataset['misclassified4'] * dataset['prob4']) 
    # calculation of performance (alpha)
    alpha4 = 0.5*log((1-e4)/e4)
    
    new_weight4 = dataset['prob4']*np.exp(-1*alpha4*dataset['target']*dataset['pred4'])
    z = sum(new_weight4)
    normalized_weight4 = new_weight4/sum(new_weight4)
    dataset['final_weights'] = round(normalized_weight,4)
    Final_weights = dataset['final_weights']
    
    return Final_weights


print("===================Best_cluster_selction=====================")

def best_cluster(x_features, labels):
    x_train,x_test,y_train,y_test=train_test_split(x_features,labels, train_size=0.8, test_size=0.2, random_state= 0)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
    x_train, x_miss, y_train, y_miss = train_test_split(x_train, y_train, test_size=0.25, random_state=3)
    
    total_enteries = np.size(x_miss)
    missing_percentage = int(0.2*total_enteries)   #  20% of data will be missed

    x_missing = x_miss.values
    for i in range(missing_percentage):
        x = random.randint(0, x_miss.shape[0]-1)
        y = random.randint(0,x_miss.shape[1]-1)
        x_missing[x,y] = 0
    x_miss = pd.DataFrame(x_missing)
     
    cls_acc_result=[]
    final_cls=0.0
    final_acc=0.0
    for jj in range(3,6):   
        cls_num=jj
    
        train_labels , train_centers, train_data, train_membership = main_func(x_test,cls_num)
        updated_train = data_managing(x_train, train_membership)
        
        miss_labels , miss_centers, miss_data, miss_membership = main_func(x_miss,cls_num)
        updated_miss = data_managing(x_miss,miss_membership)
        updated_1111 = updated_miss.copy()
        imputed_values=replacing_values(updated_miss, updated_train)
    
        imputed_val_comparison = []    
        for k, g in enumerate(imputed_values):
            imputed_val_comparison.append(g[0:-2]) 
    
        merged_result=merging_data_zero(x_train,y_train,imputed_val_comparison,y_miss)  
        merged_result_feature=merged_result.iloc[:, 0:-1] 
        merged_result_target= merged_result.iloc[:, -1 ]
    
        model= tree.DecisionTreeClassifier()  
        model=model.fit(merged_result_feature,merged_result_target)  
        predictions_result1=model.predict(x_valid)
        acc_result=accuracy_score(y_valid,predictions_result1)
        
        cls_acc_result.append([cls_num,acc_result]) 
        if acc_result>final_acc:
            final_acc = acc_result
            final_cls = cls_num
        
    return final_cls,cls_acc_result
print("========================Euclidean_distance=========================")

def euclidean_distance(point1,point2):
    dis=0
    for i in range(len(point1)):
        dis+=(point1[i]-point2[i])**2
    return dis**0.5

print("========================Main_function=========================")

def main_func(data,clus_num):
    df = data  
    num_attr = len(df.columns) - 1
    MAX_ITER = 100     
    m = 2.00          
    k = clus_num
    n = len(df)      

    def initializeMembershipMatrix(): 
        membership_mat = list()
        for i in range(n):
            random_num_list = [random.random() for i in range(k)]
            summation = sum(random_num_list)
            temp_list = [x / summation for x in random_num_list]  
            membership_mat.append(temp_list)
        return membership_mat
   
    def calculateClusterCenter(membership_mat):
        cluster_mem_val = zip(*membership_mat)
        cluster_centers = list()
        cluster_mem_val_list = list(cluster_mem_val)
        for j in range(k):
            x = cluster_mem_val_list[j]
            xraised = [e ** m for e in x]          
            denominator = sum(xraised)
            temp_num = list()
            for i in range(n):
                data_point = list(df.iloc[i])
                #for product
                prod = [xraised[i] * val for val in data_point]               
                temp_num.append(prod)
            numerator = map(sum, zip(*temp_num))
            center = [z / denominator for z in numerator]  
            cluster_centers.append(center)
        return cluster_centers
    
    def updateMembershipValue(membership_mat, cluster_centers):     
        data = []
        for i in range(n):
            x = list(df.iloc[i]) 
            data.append(x)
            #numpy.luinearalgebra.norm for normalizing data
            distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
            for j in range(k):
                den = sum([math.pow(float(distances[j] / distances[c]), 2) for c in range(k)])
                membership_mat[i][j] = float(1 / den)
        return membership_mat, data
    
    def getClusters(membership_mat):
        cluster_labels = list()
        for i in range(n):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
            cluster_labels.append(idx)
        return cluster_labels
    
    def fuzzyCMeansClustering():
        membership_mat = initializeMembershipMatrix()
        curr = 0
        while curr <= MAX_ITER: 
            cluster_centers = calculateClusterCenter(membership_mat)
            membership_mat, data = updateMembershipValue(membership_mat, cluster_centers)
            cluster_labels = getClusters(membership_mat)
            curr += 1       
        return cluster_labels, cluster_centers, data, membership_mat
       
    def xie_beni(membership_mat, center, data):
        sum_cluster_distance = 0
        min_cluster_center_distance = inf
        for i in range(k):
            for j in range(n):
                sum_cluster_distance = sum_cluster_distance + membership_mat[j][i] ** 2 * sum(
                    power(data[j, :] - center[i, :], 2))  # 
        for i in range(k - 1):
            for j in range(i + 1, k):       
               
                cluster_center_distance = sum(power(center[i, :] - center[j, :], 2))
                if cluster_center_distance < min_cluster_center_distance:
                    min_cluster_center_distance = cluster_center_distance
        return sum_cluster_distance / (n * min_cluster_center_distance)
    
    
    labels, centers, data, membership = fuzzyCMeansClustering()
    center_array = array(centers)
    label = array(labels)
    datas = array(data)

    return labels, centers, data, membership

print("========================Data_Managing=========================")

def data_managing(data, membership):
    x_data = data.to_numpy()
   
    update_test =[]
    for i , mf in enumerate(membership):  
          m_value=0
          cl_no=0
          for j, m_j in enumerate(mf):
              
              if m_j>m_value:
                  m_value = m_j
                  cl_no = j  
              else:
                  pass
             
          update_test.append(np.append(x_data[i],[cl_no,m_value],axis=0))
          
    return update_test

print("========================Replacing_Values=========================")

def replacing_values(test, train):  
    no_of_col =  len(train[0])
    for x, m in enumerate(test):     
      
          m=[0 if math.isnan(x) else x for x in m]  
          m = np.array(m)
          for i in range(m.shape[0]-2):   
            
              if m[i]==0:
                  entry = i      
                  cluster = m[no_of_col-2]   
                  mem_value = m[no_of_col-1]              
                  list_t=[]
                  for k, m_m in enumerate(train):  
                      if m_m[no_of_col-2]==cluster:       
                          list_t.append(m_m)
                                               
                  avg = 0
                  avg_x=0
                  den = 0.000001
                  for kk, xx in enumerate(list_t):
                      avg=avg+xx[entry]          
                      den=den+1.0               
                      
                  avg_x = avg/(den)                               
                  test[x][entry]=avg_x     
              else:
                  pass

    return  test

print("========================Merging Data=========================")

def merging_data_zero(x_train,y_train,imputed_val_comparison,y_test):     
    splited_train=x_train
    train_labeled=x_train.values.tolist()  
    
    train_target_values =[]
    for m, n in enumerate(y_train):
        train_target_values.append(np.append(train_labeled[m],[n],axis=0))
        
    imputed_final_test=imputed_val_comparison   
    imputed_target_values =[]
    for m, n in enumerate(y_test):
        imputed_target_values.append(np.append(imputed_final_test[m],[n],axis=0))
        
    merged_result=train_target_values + imputed_target_values
    merged_result=pd.DataFrame(merged_result)   
    
    return merged_result

print("========================Merging Data=========================")

def merging_data_nan(x_train,y_train,nan_merged_clmf_feature,nan_merged_clmf_target):
    train_merged_result= pd.concat([x_train,y_train], axis=1) 
    nan_merged_result = pd.concat([nan_merged_clmf_feature,nan_merged_clmf_target], axis=1)
    train_merged_result = np.array(train_merged_result) 
    nan_merged_result  = np.array(nan_merged_result)    
    combined_result = np.concatenate((train_merged_result, nan_merged_result), axis=0)
    combined_result = pd.DataFrame(combined_result)       
    
    return combined_result

print("========================Models_fitting=========================")

def models_fitting(x_train, y_train, x_valid,y_valid):
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    results = []
    names = []
    acc_result = []
        
    for name, model in models:
        models=model.fit(x_train, y_train) 
        predictions_result=models.predict(x_valid) 
        results.append(predictions_result) 
        names.append(name)
        acc_score=accuracy_score(y_valid,predictions_result) 
        acc_result.append(acc_score)
        
    return names,acc_result

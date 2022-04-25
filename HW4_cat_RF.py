#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:39:05 2022
"""
import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

################################
# This algorithm makes use of 3 main classes: A Decision Tree class the contains the 
# primary functions needed to building the tree, A node class to keep track of the branches
# that have been created, and a Leaf function that classifies the instances.   
class DecisionTree:
    def __init__(self, data, nTree):
        data.columns = data.columns.str.strip()
        self.bootstraps = {T: data.sample(n=len(data), replace = True, random_state=1) for T in range(nTree)}

#splits the data by the best attribute
    def response_splits(self, D, attribute):  
        no_vote = D.loc[D[attribute] == 0]
        vote_yea = D.loc[D[attribute] == 1]
        vote_nay = D.loc[D[attribute] == 2]
        return no_vote, vote_yea, vote_nay

# Computes the entropy given some labels 
    def entropy(self, labels): 
        total_rows = len(labels)
        probabilities = [count/total_rows for count in Counter(labels).values()]
        entropy = stats.entropy(probabilities, base=2)
        return entropy

# The next two functions are used to compute the information gain of a potential split   
    def weighted_avg(self, D, no_vote, vote_yea, vote_nay):    
        branch_average = []
        if len(no_vote) != 0:
            avgs = (len(no_vote)/len(D))*self.entropy(no_vote.iloc[:,-1])
            branch_average.append(avgs)
        if len(vote_yea) != 0:
            avgs = (len(vote_yea)/len(D))*self.entropy(vote_yea.iloc[:,-1])
            branch_average.append(avgs)
        if len(vote_nay) != 0:
            avgs = (len(vote_nay)/len(D))*self.entropy(vote_nay.iloc[:,-1])
            branch_average.append(avgs)
        partition_entropy = sum(branch_average)
        return partition_entropy

    def IG(self, D, partition_entropy):
        overall_entropy = self.entropy(D.loc[:,'target'])
        return overall_entropy - partition_entropy

# This functions loops through m random attributes and computes information gain of each possible split. 
# Only the attribute that led to the highest gain is selected
    def select_attribute(self, D):
        highest_gain = 0
        best_attribute = 0
        testable_attributes = list(D.columns[D.columns != 'target']) #get list of testable attributes, except for target
        m = np.ceil(np.sqrt(len(testable_attributes)))
        sample_attributes = np.random.choice(testable_attributes, int(m), replace = False)
        for i in range(len(sample_attributes)):
                no_vote, vote_yea, vote_nay = self.response_splits(D, sample_attributes[i]) 
                partition_entropy = self.weighted_avg(D, no_vote, vote_yea, vote_nay)
                information_gain = self.IG(D, partition_entropy) 
                if information_gain > highest_gain:
                    highest_gain = information_gain 
                    best_attribute = sample_attributes[i]
        return highest_gain, best_attribute

# This is the recurcive part of the tree. It takes in some dataframe, and first determines the majority_class
# for cases a split results in an empty dataset or there are no more attributes to test.
# Then it identifies whether a leaf case applied by checking the length of the dataset (<3 instances), whether all of the instances are of the 
# same class, or there are no more attributes to test. It also returns a leaf if the information gain is less than 0.05. 
# Otherwise, it recursively builds each branch until a leaf is returned.
    def grow_branches(self, D, majority_class): 
        if len(D) <= 3 or len(np.unique(D.iloc[:,-1])) == 1 or D.shape[1] == 1:
            return Leaf(D, majority_class)
        
        highest_gain, best_attribute = self.select_attribute(D)
        if highest_gain <= 0.05:
            return Leaf(D, majority_class)
        else:
            no_vote, vote_yea, vote_nay = self.response_splits(D, best_attribute)
            no_resp_branch = self.grow_branches(no_vote, majority_class)
            yea_branch = self.grow_branches(vote_yea, majority_class)
            nay_branch = self.grow_branches(vote_nay, majority_class)
        return Node(best_attribute, no_resp_branch, yea_branch, nay_branch)

# Create a leaf class to hold all of our leaves
class Leaf:
    def __init__(self, D, majority_class):
        if len(D) == 0:
            self.predicted_classes = majority_class[0]
        else:
            self.predicted_classes = D.iloc[:,-1].value_counts().idxmax()

# Create a decision node class to keep track of the partitions we make 
class Node:
    def __init__(self, best_attribute, no_resp_branch, yea_branch, nay_branch):
        self.node_attribute = best_attribute
        self.no_resp_branch = no_resp_branch
        self.yea_branch = yea_branch
        self.nay_branch = nay_branch

# This function is also recursive and works by using the trained tree to determine what path a 
# new instance should take. It will first look at an attribute, then look at the new instances response for that attribute, and follow
# that branch. This occurs until it reaches a leaf.
def predict_new_instance(new_data, trained_tree):
    if isinstance(trained_tree, Leaf):
        return trained_tree.predicted_classes
    if new_data[trained_tree.node_attribute] == 0:
        return predict_new_instance(new_data, trained_tree.no_resp_branch)
    elif new_data[trained_tree.node_attribute] == 1:
        return predict_new_instance(new_data, trained_tree.yea_branch)
    else:
        new_data[trained_tree.node_attribute] == 2
        return predict_new_instance(new_data, trained_tree.nay_branch)

# The fuction creates a stratified kfold split. it creates k subsets that are proportional in class representation to the full dataset. 
def stratified_kfold(dataset, folds):
    dataset_split = []
    c1 = dataset[dataset["target"] == 0] 
    c2 = dataset[dataset["target"] == 1]
    c1_len = len(c1)
    c2_len = len(c2)
    for i in range(folds):
        fold = []
        if i == 9:
            fold.append(c1)
        else:
            r = c1.sample(int(np.round(c1_len/folds)))
            idx = r.index
            fold.append(c1.loc[idx])
            c1 = c1.drop(idx)
        if i == 9:
            fold.append(c2)
        else:
            r = c2.sample(int(np.round(c2_len/folds)))
            idx = r.index
            fold.append(c2.loc[idx])
            c2 = c2.drop(idx)
        dataset_split.append(pd.concat(fold).sample(frac = 1))
    return dataset_split   

# this function returns that accuracy, precision, recall and F1 ratio
def performance_metrics(predicted_class, true_class, beta):
    TN, FP, FN, TP = confusion_matrix(true_class.iloc[:,-1], predicted_class).ravel()
    accuracy = (TP+TN)/(TN+FP+FN+TP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = (1+beta**2)*((precision*recall)/((beta**2)*(precision+recall)))
    return [accuracy, precision, recall, F1]

# this function actually makes the predictions given a new dataset
def predictions(new_data, trained_tree):
    predicted_labels = []
    for i in range(len(new_data)):
        predict_label = predict_new_instance(new_data.iloc[i,:], trained_tree)
        predicted_labels.append(predict_label)
    return predicted_labels

# once all of the predictions are made for every tree in the fold, then we want to choose the 
# class that was predicted most often across the trees. 
def max_predicted_class(predictions):   
    length = max(map(len, predictions))
    adj_predictions = np.array([xi+[None]*(length-len(xi)) for xi in predictions])
    combined_predictions = []
    for i in range(length):
        combined_predictions.append(np.bincount(adj_predictions[:,i]).argmax())
    return combined_predictions
   
################################
# Read in the data
gov = pd.read_csv('https://people.cs.umass.edu/~bsilva/courses/CMPSCI_589/Spring2022/homeworks/datasets/house_votes_84.csv', header = 0)     

nTree = [1,5,10,20,30,40,50] #define how many trees we want
overall_accuracy_by_nTree = []

# This implements that actual algorithm 
for n in nTree:
    folds = stratified_kfold(gov, 10) #creates 10 folds
    for fold in range(len(folds)):
        foldwise_accuracy = []
        test_fold = folds[fold] #define which fold is the test fold
        training = pd.concat(folds[:fold] + folds[fold+1:])
        DT_model = DecisionTree(training, n)
        print("new fold")
        for b in DT_model.bootstraps:
            print("new tree growing")
            ntree_predicted = []
            bootstrap = DT_model.bootstraps[b]
            majority_class = bootstrap.iloc[:,-1].mode()
            tree = DT_model.grow_branches(bootstrap, majority_class)
            ntree_predicted.append(predictions(test_fold, tree))
        predictions_across_trees = max_predicted_class(ntree_predicted)
        foldwise_accuracy.append(performance_metrics(predictions_across_trees, test_fold, 1))
    overall_accuracy_by_nTree.append(foldwise_accuracy)

#figure for ACCURACY
acc = [item[0][0] for item in overall_accuracy_by_nTree]
plt.bar(range(7), acc, color  = 'g')
plt.xticks(range(7),[1,5,10,20,30,40,50])
plt.title('ACCURACY by nTree in Random Forest')
plt.ylabel('Accuracy')
plt.xlabel('nTree')

#figure for PRECISION
p = [item[0][1] for item in overall_accuracy_by_nTree]
plt.bar(range(7), p, color  = 'g')
plt.xticks(range(7),[1,5,10,20,30,40,50])
plt.title('PRECISION by nTree in Random Forest')
plt.ylabel('precision')
plt.xlabel('nTree')

#figure for RECALL
r = [item[0][2] for item in overall_accuracy_by_nTree]
plt.bar(range(7), r, color  = 'g')
plt.xticks(range(7),[1,5,10,20,30,40,50])
plt.title('RECALL by nTree in Random Forest')
plt.ylabel('recall')
plt.xlabel('nTree')

#figure for F1
f1 = [item[0][3] for item in overall_accuracy_by_nTree]
plt.bar(range(7), f1, color  = 'g')
plt.xticks(range(7),[1,5,10,20,30,40,50])
plt.title('F1 RATIO by nTree in Random Forest')
plt.ylabel('F1 ratio')
plt.xlabel('nTree')

# This code was adopted from github and was only used for debugging purposes so that the tree is
# growing as expected  
#def print_tree(node, spacing=""):
#    if isinstance(node, Leaf):
#        print (spacing + "Predict", node.predicted_classes)
#        return
#    print (spacing + str(node.node_attribute))
#    
#    print (spacing + '--> nan:')
#    print_tree(node.no_resp_branch, spacing + "  ")
#    
#    print (spacing + '--> yea:')
#    print_tree(node.yea_branch, spacing + "  ")
#    
#    print (spacing + '--> nay:')
#    print_tree(node.nay_branch, spacing + "  ")
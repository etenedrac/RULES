
#########################
#        IMPORTS        #
#########################
import pandas as pd
import time
from itertools import combinations, product
from sklearn.model_selection import KFold
import numpy as np
from RULES.rules_trn import RULES
from RULES.rules_tst import RULES_test, RULES_test_2, RULES_test_3, RULES_test_4


#########################
#        METRICS        #
#########################

def RULES_print(rule, index):
    """
    This function prints a given rule.
    """
    print('{indx}. IF '.format(indx=index),end='')
    for j, (key,value) in enumerate(rule[0].items()):
        if j==len(rule[0])-1:
            print("{attr} is {val} THEN ".format(attr=key, val=value), end='')
        else:
            print("{attr} is {val} AND ".format(attr=key, val=value), end='')
    print("Class is {cls}.".format(cls=rule[1]))

def RULES_metrics(df, class_atr, rules, verbose=True):
    """
    This function computes the precision and coverage of a given rule set
    on a dataset. It optionally prints the rules in a pretty format with
    the coverage ans precision.
    """
    
    total_length = len(df)
    precision = []
    coverage = []
    
    for index, rule in enumerate(rules):
        df2 = df.copy()
        #Getting the features and values for a given rule
        features = list(rule[0].keys())
        values = list(rule[0].values())
        rl = zip(features,values)
        
        i = 0
        for key, val in rl:
            if(i==0):
                a = df2[key]==val
            else:
                b = df2[key]==val
                a = a & b
            i = i+1
        keep = a
        df2 = df2[keep]
        length_covered = len(df2)
        cov = length_covered/total_length
        coverage.append(cov)
        
        #Measure the precision
        c = df2[class_atr]==rule[1]
        df2 = df2[c]
        length_correct = len(df2)
        prec = length_correct/length_covered
        precision.append(prec)
        
        #Print the rule and metric
        if verbose:
            RULES_print(rule,index+1)
            print("\tPrecision: ", prec)
            print("\tCoverage: ", cov)
        
    return precision, coverage

def prevent_conflicts_rules(df, cls_atr):
    """
    This method eliminates duplicated instances in a dataset
    which have a different class atribute.
    """
    df2 = df.copy()
    del df2[cls_atr]
    df2 = df2[~df2.duplicated()]
    df = df.iloc[df2.index.values]
    return df



#########################
#      EVALUATIONS      #
#########################

def RULES_evaluate(df, cls_atr):
    """
    This method prints all the rules for a given dataset
    with the corresponding metrics.
    """
    
    df = prevent_conflicts_rules(df, cls_atr)
    
    #Train
    final_rules = RULES(df, cls_atr)
    
    #Show results
    RULES_metrics(df, cls_atr, final_rules)
    
    return final_rules

def RULES_k_fold_evaluate(df, cls_atr, k=3):
    """
    Evaluate correctly if any rule matches the correct class
    """
    
    kf = KFold(n_splits = k, shuffle = True, random_state = 2)
    
    accuracy = []
    elapsed = []
    
    #Getting the different train and test datasets
    for trn,tst in kf.split(df):
        train, test = df.iloc[trn], df.iloc[tst]
        
        #Prevent conflicts on the training set
        train = prevent_conflicts_rules(train.reset_index(drop=True), cls_atr)
        
        #Train
        t = time.time()
        final_rules = RULES(train, cls_atr)
        elapsed.append(time.time() - t)
        
        #Test
        accuracy.append(RULES_test(test,cls_atr,final_rules))
    mean_accuracy = np.mean(accuracy)
    std_accuracy = np.std(accuracy)
    mean_time = np.mean(elapsed)
    std_time = np.std(elapsed)
    print("Final accuracy: {mn_acc}".format(mn_acc=mean_accuracy)+u"\u00B1"
        +"{std_acc}".format(std_acc=std_accuracy))
    print("Final time: {mn_tm}".format(mn_tm=mean_time)+u"\u00B1"
        +"{std_tm}".format(std_tm=std_time))
    
def RULES_k_fold_evaluate_2(df, cls_atr, k=3):
    """
    Evaluate correctly if all the rules matches the correct class
    """
    
    kf = KFold(n_splits = k, shuffle = True, random_state = 2)
    
    accuracy = []
    elapsed = []
    
    #Getting the different train and test datasets
    for trn,tst in kf.split(df):
        train, test = df.iloc[trn], df.iloc[tst]
        
        #Prevent conflicts on the training set
        train = prevent_conflicts_rules(train.reset_index(drop=True), cls_atr)
        
        #Train
        t = time.time()
        final_rules = RULES(train, cls_atr)
        elapsed.append(time.time() - t)
        
        #Test
        accuracy.append(RULES_test_2(test,cls_atr,final_rules))
    mean_accuracy = np.mean(accuracy)
    std_accuracy = np.std(accuracy)
    mean_time = np.mean(elapsed)
    std_time = np.std(elapsed)
    print("Final accuracy: {mn_acc}".format(mn_acc=mean_accuracy)+u"\u00B1"
        +"{std_acc}".format(std_acc=std_accuracy))
    print("Final time: {mn_tm}".format(mn_tm=mean_time)+u"\u00B1"
        +"{std_tm}".format(std_tm=std_time))
    
def RULES_k_fold_evaluate_3(df, cls_atr, k=3):
    """
    Evaluate correctly if the majority of rules matches the correct class
    """
    
    kf = KFold(n_splits = k, shuffle = True, random_state = 2)
    
    accuracy = []
    elapsed = []
    
    #Getting the different train and test datasets
    for trn,tst in kf.split(df):
        train, test = df.iloc[trn], df.iloc[tst]
        
        #Prevent conflicts on the training set
        train = prevent_conflicts_rules(train.reset_index(drop=True), cls_atr)
        
        #Train
        t = time.time()
        final_rules = RULES(train, cls_atr)
        elapsed.append(time.time() - t)
        
        #Test
        accuracy.append(RULES_test_3(test,cls_atr,final_rules))
    mean_accuracy = np.mean(accuracy)
    std_accuracy = np.std(accuracy)
    mean_time = np.mean(elapsed)
    std_time = np.std(elapsed)
    print("Final accuracy: {mn_acc}".format(mn_acc=mean_accuracy)+u"\u00B1"
        +"{std_acc}".format(std_acc=std_accuracy))
    print("Final time: {mn_tm}".format(mn_tm=mean_time)+u"\u00B1"
        +"{std_tm}".format(std_tm=std_time))
    
def RULES_k_fold_evaluate_4(df, cls_atr, k=3):
    """
    Evaluate correctly if the majority of rules matches the correct class
    wheighted by the coverage of each rule.
    """
    
    kf = KFold(n_splits = k, shuffle = True, random_state = 2)
    
    accuracy = []
    elapsed = []
    
    #Getting the different train and test datasets
    for trn,tst in kf.split(df):
        train, test = df.iloc[trn], df.iloc[tst]
        
        #Prevent conflicts on the training set
        train = prevent_conflicts_rules(train.reset_index(drop=True), cls_atr)
        
        #Train
        t = time.time()
        final_rules = RULES(train, cls_atr)
        elapsed.append(time.time() - t)
        
        #Get the total classes
        total_clases = list(set(train[cls_atr].values))
        
        #Get the coverage of each rule
        precision, coverage = RULES_metrics(df, cls_atr, final_rules,verbose=False)
        
        #Test
        accuracy.append(RULES_test_4(test,cls_atr,final_rules, coverage, total_clases))
    mean_accuracy = np.mean(accuracy)
    std_accuracy = np.std(accuracy)
    mean_time = np.mean(elapsed)
    std_time = np.std(elapsed)
    print("Final accuracy: {mn_acc}".format(mn_acc=mean_accuracy)+u"\u00B1"
        +"{std_acc}".format(std_acc=std_accuracy))
    print("Final time: {mn_tm}".format(mn_tm=mean_time)+u"\u00B1"
        +"{std_tm}".format(std_tm=std_time))
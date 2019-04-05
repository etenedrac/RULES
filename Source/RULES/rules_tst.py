
#########################
#        IMPORTS        #
#########################
import pandas as pd
from itertools import combinations, product
from sklearn.model_selection import KFold
import numpy as np


#########################
#         TESTS         #
#########################

def RULES_test(test_df, class_atr, rules):
    """
    Evaluate correctly if any rule matches the correct class
    """
    df2 = test_df.copy()
    total_length = len(test_df)
    for rule in rules:
        #Getting the features and values for a given rule
        features = list(rule[0].keys())
        features.append(class_atr)
        values = list(rule[0].values())
        values.append(rule[1])
        rl = zip(features,values)
        
        i = 0
        for key, val in rl:
            if(i==0):
                a = df2[key]==val
            else:
                b = df2[key]==val
                a = a & b
            i = i+1
        keep = ~a
        df2 = df2[keep]
    final_length = len(df2)
    accuracy = 1-(final_length/total_length)
    return accuracy

def RULES_test_2(test_df, class_atr, rules):
    """
    Evaluate correctly if all the rules matches the correct class
    """
    df2 = test_df.copy()
    total_length = len(df2)
    accuracy = 0
    for index, row in df2.iterrows():
        #First we get the rules that apply to a given instance
        cls = []
        for rule in rules:
            good_rule = True
            for key, val in rule[0].items():
                if not row[key]==val:
                    good_rule = False
            if good_rule:
                cls.append(rule[1])
        #print(cls)
        #print(row[class_atr])
        #print()
        
        #Then we look if all the classes are the same:
        if len(set(cls))==1:
            #And if the hace correctly predicted the class:
            if cls[0]==row[class_atr]:
                accuracy = accuracy+1
    acc = accuracy/total_length
    return acc

def RULES_test_3(test_df, class_atr, rules):
    """
    Evaluate correctly if the majority of rules matches the correct class
    """
    df2 = test_df.copy()
    total_length = len(df2)
    accuracy = 0
    for index, row in df2.iterrows():
        #First we get the rules that apply to a given instance
        cls = []
        for rule in rules:
            good_rule = True
            for key, val in rule[0].items():
                if not row[key]==val:
                    good_rule = False
            if good_rule:
                cls.append(rule[1])
        #print(cls)
        #print(row[class_atr])
        #print()
        
        #Get the most voted rule class as the prediction:
        if most_frequent(cls)==row[class_atr]:
            accuracy = accuracy+1
    acc = accuracy/total_length
    return acc

def most_frequent(List): 
    if len(set(List))>0:
        return max(set(List), key = List.count)
    else:
        return None

def RULES_test_4(test_df, class_atr, rules, coverage, total_classes):
    """
    Evaluate correctly if the majority of rules matches the correct class
    wheighted by the coverage of each rule.
    """
    df2 = test_df.copy()
    total_length = len(df2)
    accuracy = 0
    for index, row in df2.iterrows():
        #Generate initial dictionary of weights
        dic = {}
        for elem in total_classes:
            dic.update({elem:0})
            
        #First we get the rules that apply to a given instance
        for index, rule in enumerate(rules):
            good_rule = True
            for key, val in rule[0].items():
                if not row[key]==val:
                    good_rule = False
            if good_rule:
                dic.update({rule[1]:dic[rule[1]]+coverage[index]})
        #print(cls)
        #print(row[class_atr])
        #print()
        
        #Get the most voted rule class as the prediction:
        if max(dic, key=dic.get)==row[class_atr]:
            accuracy = accuracy+1
    acc = accuracy/total_length
    return acc
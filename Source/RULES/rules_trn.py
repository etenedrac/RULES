
#########################
#        IMPORTS        #
#########################
import pandas as pd
from itertools import combinations, product
from sklearn.model_selection import KFold
import numpy as np


#########################
#         RULES         #
#########################

def RULES(dataframe, class_atr):
    #Copy the dataframe
    df = dataframe.copy()
    
    # Compute the maximum amount of conditions
    max_conditions = len(df.columns)-1
    
    #Set to 0 the length of the initial conditions
    conditions = 0
    
    all_examples_are_classified = False
    
    final_rules = []
    
    #Loop while not all examples are classified 
    while((not all_examples_are_classified) and (conditions<=max_conditions)):

        #Add another level of conditions
        conditions = conditions + 1
        
        #Get initial rules
        candidate_rules = get_candidate_rules(df, conditions, class_atr)

        #Examine if we can create a rule
        true_rules = []
        for rule in candidate_rules:
            is_good_rule, cls = rule_tester(rule, dataframe, class_atr)
            if is_good_rule:
                true_rules.append([rule,cls])
        
        #Check for irrelevant conditions
        if conditions>1:
            true_rules = relevant_conditions(true_rules, final_rules)

        #Get the non-classified instances
        df = get_unclassified_instances(true_rules,df)
        
        #Extend the rules
        final_rules.extend(true_rules)

        if(len(df)==0):
            all_examples_are_classified=True
            
    #If we have not classified all the instances, create more rules
    if(not all_examples_are_classified):
        final_rules.extend(generate_last_rules(df, class_atr))
    
    return final_rules




#########################
#         EXTRA         #
#########################

def get_candidate_rules(df, length_rules, class_atr):
    """
    Function ot generate candidate rules for the RULES algorithm.
        - df {pd.dataframe} dataframe to get the rules
        - length_rules {integer} the length of the rules
        - class_atr {string} name of the class variable
    """
    #Copy the dataset without the class atribute
    df2 = df.copy()
    del df2[class_atr]
    
    rules = []
    
    #First we get all the possible values per attribute
    possible_values = []
    for col in df2.columns:
        col_pivot = []
        for val in set(df2[col]):
            col_pivot.append({col:val})
        possible_values.append(col_pivot)
    
    #Get all the possible combinations of columns to pick rules
    combination_columns = list(combinations(range(len(possible_values)),length_rules))
    
    #For every combination of columns, generate the cartesian product
    #of its values
    for columns in combination_columns:
        
        #Get the columns associated to a given combination
        pivot_columns = []
        for column in columns:
            pivot_columns.append(possible_values[column])
            
        #Generate the rules of the combination of columns
        for element in product(*pivot_columns):
            pivot = {}
            for element_rule in element:
                pivot.update(element_rule)
            rules.append(pivot)
            
    return rules
            

def rule_tester(rule, df, class_atr):
    """
    Function to test if a given rule correctly classifies
    a given dataset.
        - rule {{features:values}} dictionary of the rule
        - df {pd.dataframe} dataframe to test the rule
        - class_atr {string} name of the class variable
    """
    #Getting the features and values for a given rule
    features = list(rule.keys())
    values = list(rule.values())
    rl = zip(features,values)
    
    #Generating a copy of the dataframe
    df2 = df.copy()
    
    #Getting the instances that fullfill the rule
    for key, val in rl:
        df2 = df2[df2[key]==val]
        
    if(len(set(df2[class_atr]))==1):
        return True, list(set(df2[class_atr]))[0]
    else:
        return False, list(set(df2[class_atr]))
    
def relevant_conditions(new_rules, old_rules):
    """
    Given a set of new rules, check if any of the old rules
    is a subset of a new generated rule, thus discarding irrelevant
    rules.
    """
    relevant_rules = []
    for i in range(len(new_rules)):
        is_irrelevant = False
        for j in range(len(old_rules)):
            if old_rules[j][0].items() <= new_rules[i][0].items():
                is_irrelevant = True
        if not is_irrelevant:
            relevant_rules.append(new_rules[i])
    return relevant_rules
                
    
def get_unclassified_instances(rules,df):
    """
    Given a set of rules and a dataframe, this function
    returns the instances that do not fullfill any rule.
    """
    df2 = df.copy()
    for rule in rules:
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
        keep = ~a
        df2 = df2[keep]
    return df2
    
def generate_last_rules(df, class_atr):
    """
    This function generates a rule for each row in a given dataframe.
        - df {dataframe} dataframe from which rules will be extracted
        - class_atr {string} name of the class atribute in the dataframe
    """
    columns = list(df.columns)

    rules = []
    for row in range(df.shape[0]):
        piv_dict = {}
        for i in range(len(columns)):
            if columns[i]==class_atr:
                continue
            else:
                piv_dict.update({columns[i]:df.iloc[row,i]})
        rules.append([piv_dict,list(df[class_atr])[row]])
    return rules
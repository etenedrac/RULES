
#########################
#        IMPORTS        #
#########################
import pandas as pd
import argparse
import time

from RULES.evaluation import RULES_evaluate
from RULES.evaluation import RULES_k_fold_evaluate
from RULES.evaluation import RULES_k_fold_evaluate_2
from RULES.evaluation import RULES_k_fold_evaluate_3
from RULES.evaluation import RULES_k_fold_evaluate_4


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="Dataset used to study located in the 'Data' folder (ended with .csv)",type=str, default="test.csv")
parser.add_argument('--classname', help="Name of the class of the dataset",type=str, default="Class")
parser.add_argument('--validate', help="Validate using k-fold CV (Default)", dest='val', action='store_true')
parser.add_argument('--no-validate', help="Do not do the validation", dest='val', action='store_false')
parser.add_argument('--k', help="Number of folds for validation (integer / Default: 3)", type=int, default=3)
parser.set_defaults(val=True)
args, unknown = parser.parse_known_args()

path = 'Data/' + args.dataset

#Read the dataset
df = pd.read_csv(path)

print('Computing the rules for all the dataset...')
print('-----------------------------------------------------------------------------')

RULES_evaluate(df, args.classname)


print('-----------------------------------------------------------------------------')
if args.val:
    elapsed = []
    print('Computing the CV using any voting (this is cheating, just to show the false results)...')
    RULES_k_fold_evaluate(df, args.classname, k=args.k)
    print('-----------------------------------------------------------------------------')
    print('Computing the CV using unanimity voting...')
    RULES_k_fold_evaluate_2(df, args.classname, k=args.k)
    print('-----------------------------------------------------------------------------')
    print('Computing the CV using majority voting...')
    RULES_k_fold_evaluate_3(df, args.classname, k=args.k)
    print('-----------------------------------------------------------------------------')
    print('Computing the CV using majority weighted by coverage voting...')
    RULES_k_fold_evaluate_4(df, args.classname, k=args.k)
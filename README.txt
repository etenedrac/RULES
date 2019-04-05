RULES algorithm, by Albert Cardenete
------------------------------------

The structure of this project is as follows:
	PW1-SEL-1819-AlbertCardenete/			->The root of the project
	├── Data/
	│   ├── breast_cancer.csv			->Dataset 1
	│   ├── cheese.csv				->Dataset for testing
	│   ├── nursery.csv				->Dataset 2
	│   ├── test.csv				->Default dataset for testing
	│   └── voting.csv				->Dataset 3
	├── Documentation/
	│   ├── cancer.txt				->Final rules for Dataset 1
	│   ├── nursery.txt				->Final rules for Dataset 2
	│   ├── voting.txt				->Final rules for Dataset 3
	│   └── REPORT_ALBERT_CARDENETE.pdf		->REPORT
	└── Source/
    		├── RULES/
		│   ├── __init__.py			->Dummy file to informe that this is a package
		│   ├── evaluation.py			->Contains functions of the evaluations and k-folds (a frontend)
		│   ├── rules_trn.py			->Contains the RULES algorithm
		│   └── rules_tst.py			->Contains the functions to test and predict
    		└── __main__.py				->Main function of the program to evaulate a given dataset
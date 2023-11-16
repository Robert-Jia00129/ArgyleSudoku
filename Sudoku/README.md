# Argyle_Sudoku
> This project was partly based on the code from [z3-sudoku](https://github.com/awkwardbunny/z3-sudoku)

This project uses python z3-solver to solve classic sudokus and argyle sudokus using various techniques. It then compares the efficiency of each method and between the sudokus. 

### File descriptions: 
`main.py`: This file contains the main function. It takes in a sudoku file and solves it using the methods in sudoku.py. It then prints/ the solution and the time taken to solve it.

`Sudoku.py`: Contains all functionalities of building sudoku with various constraints, logging sudoku instances to files in string format and smt format, 

`currline.txt`: 

`hard_sudoku_instance-logFile/`: Contains the hard sudoku instances in string format. 

`smt-logFiles/`: Contains the smt files for sudokus 

`hard-smt-logFiles/`: Contains the hard suodoku instances in smt format. 

`time-record/`: Cntains the time taken to solve/generate each sudoku


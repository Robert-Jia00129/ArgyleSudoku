# Argyle_Sudoku
> This project was partly based on the code from [z3-sudoku](https://github.com/awkwardbunny/z3-sudoku)

This project uses python z3-solver to solve classic sudokus and argyle sudokus using various techniques. It then compares the efficiency of each method and between the sudokus. 

## File descriptions: 
`main.py`: This file contains the main function. It takes in a sudoku file and solves it using the methods in sudoku.py. It then prints/ the solution and the time taken to solve it.

`Sudoku.py`: Contains all functionalities of building sudoku with various constraints, logging sudoku instances to files in string format and smt format, 

`currline.txt`: stores the which line of the full sudokus file should the solver generating sudoku holes start loading from and solving when calling `run_experiment`

`hard_sudoku_instance-logFile/`: Contains the hard sudoku instances in string format. 

`smt-logFiles/`: Contains the smt files for sudokus 

`hard-smt-logFiles/`: Contains the hard suodoku instances in smt format. 

`time-record/`: Cntains the time taken to solve/generate each sudoku

### hard_sudoku_instance-logFile
`argyle_instance.txt`: contains hard argyle sudoku instances that are hard if build with certain restrictions

`argyle_time.txt`: records the time for different solvers to solve a particular argyle sudoku instance built with different constrains

`classic_instance.txt`: same as `argyle_instance.txt` but for classic instances

`classic_time.txt`: same as `argyle_time.txt` but records time for solvers to solve classic instances

`curr_instance_line.txt`: record how many lines for instances have been read. So the time file can know where to read from next 

**Explaination for file structure of `argyle/classic_time.txt`**: 

Each line of the file is a string version of a dictionary described as follows: 
explaination of the file structure:
```python
time = 5.0
did_time_out = True
tgrid = "1231321093102930129..."
tindex = (1,2)
ttry_Val = 5
tis_sat = "sat"

dict = dict(
"problem":{
    "grid": tgrid, # string
    "index": tindex, # (int, int)
    "try_Val": ttry_Val, # int
    "is_sat": tis_sat # bool
},
constraint1:{ 
# e.g. (True,True,True,True,True)
    "smt_path": "/path/to/smt_file_1.smt",
    "z3": (time, did_time_out "answer sat unsat timeout"),
    "cvc5": (time, did_time_out),
    ...
    "other_solver": (time, did_time_out)
},
constraint2:{
# e.g. (True,False,True,True,True)
    "smt_path": "/path/to/smt_file_1.smt",
    "z3": (time, did_time_out "answer sat unsat timeout"),
    "cvc5": (time, did_time_out),
    ...
    "other_solver": (time, did_time_out)
},
... # other constraints
)
```

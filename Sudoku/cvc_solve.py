import cvc5
# from cvc5 import Kind

# def solve_smt_file(file_name):
#     # Create a solver instance
#     solver = cvc5.Solver()
#
#     # Read the SMT file
#     with open(file_name, 'r') as f:
#         smt_string = f.read()
#
#     # Parse the SMT string
#     smt_expr = solver.parseTerm(smt_string)
#
#     # Check satisfiability
#     result = solver.checkSatAssuming(smt_expr)
#
#     # Print result
#     print(result)
#
# solve_smt_file('problem.smt')
#
# if __name__ == '__main__':
#     solve_smt_file('/Users/jiazhenghao/Desktop/CodingProjects/ArgyleSudoku/Sudoku/smt-logFiles/09_28_00_27_101695878830.7237408')
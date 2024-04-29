import jz3 as z3


def solver_demo():
    solver = z3.Solver(benchmark_mode=True)

    x = z3.Int('x')
    y = z3.Int('y')

    solver.add(x > 0)
    solver.add(y > 0)

    condition1 = z3.Bool('condition1')
    condition2 = z3.Bool('condition2')

    solver.add_global_constraints(z3.Or(condition1, condition2))
    solver.add_global_constraints(z3.Distinct(condition1,condition2))

    solver.add_conditional_constraint(x < 5, condition=condition1)
    solver.add_conditional_constraint(x > 5, condition=condition2)

    solver.start_recording()
    result = solver.check_conditional_constraints()
    print(result)

    # Access the recorded combinations and performance results
    print("Condition Variable Assignment Models:")
    print(solver.get_condition_var_assignment_model())
    print("Solvers Results for each variable assignment:")
    print(solver.get_var_assignments_and_solvers_performance())

if __name__ == '__main__':
    solver_demo()
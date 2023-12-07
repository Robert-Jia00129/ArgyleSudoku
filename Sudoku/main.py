import os
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Tuple, List, Hashable
from itertools import islice
import random

import Sudoku

FULL_CONDITIONS = [(classic, distinct, percol, nonum, prefill)  # must be hashable
                   for classic in (True, False)
                   for distinct in (True, False)
                   for percol in (True, False)
                   for nonum in (True, False) if not (distinct and nonum)
                   for prefill in (True, False)]
assert isinstance(FULL_CONDITIONS[0], Hashable), "Conditions MUST be HASHABLE"
SOLVER_LIST = ("z3", "cvc5")


# Remember to reset the dict in curr_line.txt if the file exist


def write_file(condition_name, arr_time):
    file_path = condition_name + "-" + time.strftime("%Y_%m_%d_%H_%M_%S")
    with zipfile.ZipFile(f'../{file_path}.zip', 'w') as my_zip:
        files = os.listdir('.')
        for f in files:
            my_zip.write(f)
        with my_zip.open(f"{condition_name}.txt", "w") as new_hello:
            new_hello.write(bytes(f'{arr_time}', 'utf-8'))


def to_str(bool_list) -> str:
    if len(bool_list) == 6:
        return ''.join(("classic-" if bool_list[0] else "argyle-",
                        "distinct-" if bool_list[1] else "PbEq-",
                        "percol-" if bool_list[2] else "inorder-",
                        "is_bool-" if bool_list[3] else "is_num-",
                        "prefill-" if bool_list[4] else "no_prefill-",
                        "gen_time" if bool_list[5] else "solve_time"))
    if len(bool_list) == 5:
        return ''.join(("classic-" if bool_list[0] else "argyle-",
                        "distinct-" if bool_list[1] else "PbEq-",
                        "percol-" if bool_list[2] else "inorder-",
                        "is_bool-" if bool_list[3] else "is_num-",
                        "prefill-" if bool_list[4] else "no_prefill-"))


def to_bool(condition_str: str) -> list[bool]:
    condition_lst = condition_str.split('-')
    assert len(condition_lst) >= 4, "Cannot convert condition string"
    if len(condition_lst) == 4:
        return [True if condition_lst[0].lower() == "classic" else False,
                True if condition_lst[1].lower() == "distinct" else False,
                True if condition_lst[2].lower() == "percol" else False,
                True if condition_lst[3].lower() == "is_bool" else False,
                True if condition_lst[4].lower() == "prefill" else False,
                True if condition_lst[5].lower() == "gen_time" else False]
    pass


def run_experiment(single_condition: bool, *args,
                   full_iter: int = 0, holes_iter: int = 0, total_time_per_condition=5 * 60, start_condition=[],
                   end_condition=[],
                   start_from_next=False,
                   curr_line_path: str = 'curr_line.txt',
                   classic_full_path: str = '../store-sudoku/classic_full_sudokus.txt',
                   argyle_full_path: str = '../store-sudoku/argyle_full_sudokus.txt',
                   classic_holes_path: str = '../store-sudoku/classic_holes_sudokus.txt',
                   argyle_holes_path: str = '../store-sudoku/argyle_holes_sudokus.txt'):
    try:
        with open(curr_line_path, 'r') as f:
            curr_line = eval(f.readline())
    except IOError:
        curr_line = {}  # Keep track of which full sudoku to continue_reading with

    total_solve = {}
    if single_condition:
        conditions = [start_condition]
    else:
        conditions = [[classic, distinct, percol, nonum, prefill]
                      for classic in (True, False)
                      for distinct in (True, False)
                      for percol in (True, False)
                      for nonum in (True, False) if not (distinct and nonum)
                      for prefill in (True, False)]
        if start_condition:
            conditions = conditions[conditions.index(start_condition) + start_from_next:]
    if full_iter > 0:
        for ele in conditions:
            exceed_time_limit = False
            # full_sudoku_path = '../store-sudoku/' + ''.join(condition) + 'full_sudokus.txt'
            if ele[0]:
                full_sudoku_path = classic_full_path
                hard_sudoku_path = 'hard_sudoku_instance-logFile/classic_instances.txt'
            else:
                full_sudoku_path = argyle_full_path
                hard_sudoku_path = 'hard_sudoku_instance-logFile/argyle_instance.txt'

            condition_name = to_str(ele) + 'full_time'
            condition_progress = f'{conditions.index(ele) + 1}/{len(conditions)}'
            for i in range(full_iter):
                print(f'{i + 1}th iteration: Processing full sudoku {condition_name}'
                      f'Total Progress: {condition_progress} of all conditions')

                if condition_name not in total_solve:
                    total_solve[condition_name] = 0
                if total_solve[condition_name] > total_time_per_condition:
                    # record current position
                    exceed_time_limit = True
                    break
                full_time, full_penalty = Sudoku.gen_full_sudoku(*ele, hard_smt_logPath='smt-logFiles/',
                                                                 hard_sudoku_logPath=hard_sudoku_path,
                                                                 store_sudoku_path=full_sudoku_path)
                total_solve[condition_name] += full_time
                with open('../time-record/' + condition_name + '.txt',
                          'a') as f:  # if error, create ../time-record directory
                    f.write(f'{full_time},{full_penalty}\n')
            if exceed_time_limit:
                print(f'{full_sudoku_path} {ele} exceeded time limit when generating full_grid')
    if holes_iter > 0:
        for ele in conditions:
            enough_sudoku = True
            if ele[0]:
                full_sudoku_path = classic_full_path
                holes_sudoku_path = classic_holes_path
                hard_sudoku_path = 'hard_sudoku_instance-logFile/classic_instances.txt'
            else:
                full_sudoku_path = argyle_full_path
                holes_sudoku_path = argyle_holes_path
                hard_sudoku_path = 'hard_sudoku_instance-logFile/argyle_instance.txt'

            with open(full_sudoku_path, 'r') as f:
                if full_sudoku_path in curr_line:
                    f.seek(curr_line[full_sudoku_path])
                condition_name = to_str(ele) + 'holes_time'
                for i in range(holes_iter):
                    print(f'{i + 1}th iteration: Processing holes sudoku {condition_name}')
                    sudoku_lst = f.readline()[:-1]  # get rid of new line character
                    if condition_name not in total_solve:
                        total_solve[condition_name] = 0
                    if total_solve[condition_name] > total_time_per_condition:
                        enough_sudoku = False
                        break
                    # holes_time, holes_penalty = Sudoku.gen_holes_sudoku(eval(sudoku_lst), *ele,
                    # hard_instances_log_path='DataCollection/', store_sudoku_path='../store-sudoku/' + condition_name +
                    # '.txt')
                    holes_time, holes_penalty = Sudoku.gen_holes_sudoku(eval(sudoku_lst), *ele,
                                                                        hard_smt_logPath='smt-logFiles/',
                                                                        hard_sudoku_logPath=hard_sudoku_path,
                                                                        store_sudoku_path=holes_sudoku_path)
                    print(f'\tTime taken: {holes_time}')
                    total_solve[condition_name] += holes_time
                    with open('../time-record/' + condition_name + '.txt', 'a+') as f_holes:
                        f_holes.write(f'{holes_time},{holes_penalty}\n')

                curr_line[full_sudoku_path] = f.tell()
                f.read()
                file_size = f.tell()
                print(f'{curr_line[full_sudoku_path] / file_size * 100}% of the full grid for '
                      f'{full_sudoku_path.removesuffix("full_sudokus.txt")} {ele} is used')
            print(f'{["NOT ", ""][enough_sudoku]}enough sudoku for this constraint')

            par_dir = Path(curr_line_path).parent
            if not os.path.exists(par_dir):
                os.makedirs(par_dir)
            with open(curr_line_path, 'w') as f:
                f.truncate()
                f.write(str(curr_line))

        # Increament both time
    print("Process Finished")


def load_and_alternative_solve_outdated(hard_instances_file_dir: str, is_classic: bool, num_iter: int,
                                        currline_path="curr_instance_line.txt"):
    """
    Writes a dictionary with {problem: , cond_1_time: , cond_2_time: cond_3_time: cond_4_time: ...}
    Condition[0] MUST be TRUE when classic and FALSE when argyle
    :param file_path:
    :return: None
    """
    assert os.path.isdir(hard_instances_file_dir), "directory provided does not exist"
    if is_classic:
        hard_instances_file_path = hard_instances_file_dir + "classic_instances.txt"
        store_comparison_file_path = hard_instances_file_dir + "classic_time.txt"
    else:
        hard_instances_file_path = hard_instances_file_dir + "argyle_instance.txt"
        store_comparison_file_path = hard_instances_file_dir + "argyle_time.txt"

    with open(hard_instances_file_path, 'r+') as fr:  # this line not really necessary, but I'm afraid to remove
        with open(currline_path, "r") as ftempr:
            argyle_and_classic_time_dict = ftempr.readline()
            if argyle_and_classic_time_dict == '':
                argyle_and_classic_time_dict = {"classic": 0, "argyle": 0, "seed": 40}
            else:
                argyle_and_classic_time_dict = eval(argyle_and_classic_time_dict)
        curr_rand_index: int = argyle_and_classic_time_dict["classic" if is_classic else "argyle"]
        rand_seed: int = argyle_and_classic_time_dict["seed"]
        temp_iteration_num = 0
        num_lines = sum(1 for _ in open(hard_instances_file_path))  # TODO: This also needs modifications @sj
        line_generator = random_lines(hard_instances_file_path, rand_seed, curr_rand_index, num_lines)
        for _ in range(num_iter):
            line_to_solve = next(line_generator)
            try:
                sudoku_grid, condition, index, try_val, is_sat = line_to_solve.strip().split('\t')
            except ValueError:
                continue
            store_result_dict = {}
            sudoku_grid = list(sudoku_grid)
            sudoku_lst = list(map(int, sudoku_grid))
            condition = eval(condition)
            index = eval(index)
            try_val = eval(try_val)
            is_sat = is_sat == "sat"
            # solve with other conditions
            store_result_dict["problem"]["grid"] = ''.join(sudoku_grid)
            store_result_dict["problem"] = index
            store_result_dict["problem"] = try_val
            store_result_dict["problem"] = is_sat
            CorAconditions = [ele for ele in FULL_CONDITIONS if ele[0] == condition[0]]
            for (i, CorAcondition) in enumerate(CorAconditions):
                time, penalty = Sudoku.check_condition_index(sudoku_lst, CorAcondition, index, try_val, is_sat)
                store_result_dict[str(CorAcondition)] = (time, penalty)
            with open(store_comparison_file_path, 'a+') as fw:
                fw.write(str(store_result_dict) + '\n')

        with open(currline_path, "w") as ftempw:
            ftempw.truncate()
            curr_rand_index += num_iter
            argyle_and_classic_time_dict["classic" if is_classic else "argyle"] = curr_rand_index
            ftempw.write(f'{argyle_and_classic_time_dict}\n')


def random_lines(filename, seed, index, num_lines):
    # TODO: This still needs modifications. @sj
    line_prng = random.Random(seed)
    line_numbers = list(range(1, num_lines + 1))
    line_prng.shuffle(line_numbers)
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i in range(index, num_lines):
        line_number = line_numbers[i]
        yield lines[line_number - 1].strip()


def solve_with_z3(smt_log_file_path: str, time_out: int) -> (int, int):
    """

    :param smt_log_file_path:
    :param time_out: in milliseconds
    :return:
    """
    start_time = time.time()
    result = subprocess.run(["z3", "-smt2", smt_log_file_path, f"-T:{time_out / 1000}"], capture_output=True, text=True)
    end_time = time.time()
    combined_output = result.stdout + result.stderr  # just to make sure
    return (end_time - start_time, "timeout" in combined_output)


def solve_with_cvc5(smt_log_file_path: str, time_out: int) -> (int, int):
    start_time = time.time()
    result = subprocess.run(["./cvc5-macOS-arm64", smt_log_file_path, "--lang", "smt2", "--tlimit", str(time_out)],
                            capture_output=True, text=True)
    end_time = time.time()
    combined_output = result.stdout + result.stderr  # just to make sure
    return (end_time - start_time, "cvc5 interrupted by timeout" in combined_output)


def load_and_alternative_solve(hard_instances_file_dir: str, is_classic: bool, num_iter: int,
                               currline_path="curr_instance_line.txt"):
    """
    Writes a dictionary with {problem: , cond_1_time: , cond_2_time: cond_3_time: cond_4_time: ...}
    Condition[0] MUST be TRUE when classic and FALSE when argyle
    :param file_path:
    :return: None
    """
    assert os.path.isdir(hard_instances_file_dir), "directory provided does not exist"
    if is_classic:
        hard_instances_file_path = hard_instances_file_dir + "classic_instances.txt"
        store_comparison_file_path = hard_instances_file_dir + "classic_time.txt"
    else:
        hard_instances_file_path = hard_instances_file_dir + "argyle_instance.txt"
        store_comparison_file_path = hard_instances_file_dir + "argyle_time.txt"

    with open(hard_instances_file_path, 'r+') as fr:  # this line not really necessary, but I'm afraid to remove
        with open(currline_path, "r") as ftempr:
            argyle_and_classic_time_dict = ftempr.readline()
            if argyle_and_classic_time_dict == '':
                argyle_and_classic_time_dict = {"classic": 0, "argyle": 0, "seed": 40}
            else:
                argyle_and_classic_time_dict = eval(argyle_and_classic_time_dict)
    curr_rand_index: int = argyle_and_classic_time_dict["classic" if is_classic else "argyle"]
    rand_seed: int = argyle_and_classic_time_dict["seed"]
    num_lines = sum(1 for _ in open(hard_instances_file_path))  # TODO: This also needs modifications @sj
    argyle_and_classic_time_dict[
        "classic" if is_classic else "argyle"] += curr_rand_index + num_iter  # record read lines up till now
    line_generator = random_lines(hard_instances_file_path, rand_seed, curr_rand_index, num_lines)
    for _ in range(num_iter):
        line_to_solve = next(line_generator)
        try:
            sudoku_grid, condition, index, try_val, is_sat = line_to_solve.strip().split('\t')
        except ValueError:
            continue
        store_result_dict = {}
        sudoku_grid = list(sudoku_grid)
        sudoku_lst = list(map(int, sudoku_grid))
        condition = eval(condition)  # doesn't really matter
        index = eval(index)
        try_val = eval(try_val)
        is_sat = is_sat == "sat"
        # solve with other conditions
        tgrid, tcondition, tindex, ttry_Val, tis_sat = line_to_solve.split("\t")
        store_result_dict["problem"] = {
            "grid": tgrid,
            "index": eval(tindex),
            "try_Val": eval(ttry_Val),
            "is_sat": tis_sat == "sat"
        }
        CorAconditions = [ele for ele in FULL_CONDITIONS if ele[0] == condition[0]]
        for CorAcondition in CorAconditions:
            if (CorAcondition) not in store_result_dict:
                store_result_dict[CorAcondition] = {}  # initialize the dictionary
            if "smt_path" not in store_result_dict[CorAcondition]:
                single_condition_smt_path = Sudoku.generate_smt(store_result_dict["problem"]["grid"],
                                                                CorAcondition,
                                                                store_result_dict["problem"]["index"],
                                                                store_result_dict["problem"]["try_Val"],
                                                                store_result_dict["problem"]["is_sat"],
                                                                smt_dir="smt-logFiles/")
                store_result_dict[CorAcondition]["smt_path"] = single_condition_smt_path
            else:
                single_condition_smt_path = store_result_dict["smt_path"]

            for SOLVER in SOLVER_LIST:
                store_result_dict[CorAcondition][SOLVER] = solve_with_solver(SOLVER, single_condition_smt_path)

        # write time dictionary to file
        with open(store_comparison_file_path, 'a+') as fw:
            fw.write(str(store_result_dict) + '\n')
    with open(currline_path, 'w') as fw:
        fw.truncate()
        fw.write(str(argyle_and_classic_time_dict))


def solve_with_solver(solver_name: str, smt_file_path, time_out=2) -> (int, int):
    """
    solve an smt file with particular solver
    :param solver_name:
    :param smt_file_path:
    :return: (time, did_time_out)
    """
    if solver_name == 'z3':
        return solve_with_z3(smt_file_path, time_out=time_out)
    elif solver_name == 'cvc5':
        return solve_with_cvc5(smt_file_path, time_out=time_out)
    # Add more elif blocks for other solvers
    raise ValueError(f"Unknown solver: {solver_name}, please implement the corresponding code")


if __name__ == '__main__':
    # dictionary of file paths to feed into `run_experiment`
    dct = {"curr_line_path": 'curr_line.txt',
           "classic_full_path": '../store-sudoku/classic_full_sudokus.txt',
           "argyle_full_path": '../store-sudoku/argyle_full_sudokus.txt',
           "classic_holes_path": '../store-sudoku/classic_holes_sudokus.txt',
           "argyle_holes_path": "../store-sudoku/argyle_holes_sudokus.txt"}
    # Left off with argyle-distinct-inorder-is_num-no_prefill-full_timeTotal

    hard_instances_file_dir = "hard_sudoku_instance-logFile/"
    alternative_solve_curr_line_path = "hard_sudoku_instance-logFile/curr_instance_line.txt"
    load_and_alternative_solve(hard_instances_file_dir, is_classic=True, num_iter=1,
                               currline_path=alternative_solve_curr_line_path)
    load_and_alternative_solve(hard_instances_file_dir, is_classic=False, num_iter=1,
                               currline_path=alternative_solve_curr_line_path)

    # run_experiment(False, full_iter=30, holes_iter=30,
    #                total_time_per_condition=5 * 60 * 10000000,
    #                start_condition=[True, True, False, False, False],
    #                start_from_next=True)
    # run_experiment(single_condition=False, full_iter=20, holes_iter=20,
    #                total_time_per_condition=1 * 60 * 1000)
    # run_experiment(True, [False, False, True, True, True], run_full=True, run_holes=False, full_iter=1000,
    #                total_time_per_condition = 5 * 60 * 10000000)

    print("Process Complete")
# specify timeout for python subprocesses
# don't tell time limit
# record timeout despite the output.
# exceptions 
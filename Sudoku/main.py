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


def solve_with_z3(smt_log_file_path: str, time_out: int) -> (int, int, str):
    """

    :param smt_log_file_path:
    :param time_out: in seconds
    :return:
    """
    start_time = time.time()
    did_timeout = False
    try:
        result = subprocess.run(["z3", "-smt2", smt_log_file_path],
                                capture_output=True, text=True, timeout=time_out)
        combined_output = ((result.stdout if result.stdout is not None else "") +
                           (result.stderr if result.stderr is not None else ""))  # capture all output
    except subprocess.TimeoutExpired as exc:
        did_timeout = True
        result = exc
    ans = "timeout"
    end_time = time.time()

    if not did_timeout:
        if "unsat" in combined_output:
            ans = "unsat"
        elif "sat" in combined_output:
            ans = "sat"
        else:
            ans = "unknown"
    return (end_time - start_time, did_timeout, ans)


def solve_with_cvc5(smt_log_file_path: str, time_out: int) -> (int, int, str):
    start_time = time.time()
    did_timeout = False
    try:
        result = subprocess.run(["./cvc5-macOS-arm64", smt_log_file_path, "--lang", "smt2"],
                                capture_output=True, text=True, timeout=time_out)
        combined_output = ((result.stdout if result.stdout is not None else "") +
                           (result.stderr if result.stderr is not None else ""))  # capture all output
    except subprocess.TimeoutExpired as exc:
        did_timeout = True
        combined_output = ((exc.stdout.decode('utf-8') if exc.stdout else "") +
                           (exc.stderr.decode('utf-8') if exc.stderr else ""))  # capture all output
    ans = "timeout"

    end_time = time.time()

    # TODO @sj this might not work. maybe some outputs are not in "sat" or "unsat"??
    if not did_timeout:
        if "unsat" in combined_output:
            ans = "unsat"
        elif "sat" in combined_output:
            ans = "sat"
        else:
            ans = "unknown"
    return (end_time - start_time, did_timeout, ans)


def load_and_alternative_solve(hard_instances_file_dir: str, is_classic: bool, num_iter: int,
                               currline_path="curr_instance_line.txt", timeout=5):
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
        curr_line_num: int = argyle_and_classic_time_dict.get("classic" if is_classic else "argyle", 0)
        argyle_and_classic_time_dict[
            "classic" if is_classic else "argyle"] += curr_line_num + num_iter  # record read lines up till now

        # skip current line numbers
        for _ in range(curr_line_num):
            fr.readline()

        for _ in range(num_iter):
            line_to_solve = fr.readline().strip()
            if not line_to_solve:
                print("Not enough hard instances for experiment\n\n\n\n\n\n")
            store_result_dict = {}
            try:
                tgrid, tcondition, tindex, ttry_Val, tis_sat = line_to_solve.split("\t")
            except ValueError:
                continue
            tcondition = eval(tcondition)

            # store problem and smt path
            store_result_dict["problem"] = {
                "grid": tgrid,
                "index": eval(tindex),
                "try_Val": eval(ttry_Val),
                "is_sat": tis_sat == "sat"
            }

            # solve with other conditions
            CorAconditions = [ele for ele in FULL_CONDITIONS if ele[0] == tcondition[0]]
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
                    instances_lst = store_result_dict[CorAcondition].get(SOLVER, [])
                    instances_lst.append(solve_with_solver(SOLVER, single_condition_smt_path, time_out=timeout))
                    store_result_dict[CorAcondition][SOLVER] = instances_lst

            # write time dictionary to file
            with open(store_comparison_file_path, 'a+') as fw:
                fw.write(str(store_result_dict) + '\n')
        with open(currline_path, 'w') as fw:
            fw.truncate()
            fw.write(str(argyle_and_classic_time_dict))


def solve_with_solver(solver_name: str, smt_file_path, time_out=5) -> (int, int, str):
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


def plot_constraints_comparison(option_num: int, solver: str):
    """
    Plots the comparison between option_num == True and option_num == False
    :param option_num: available options: ['classic', 'distinct', 'per_col', 'no_num', 'prefill']
    :param solver: specify which solver to use. if the string "ALL" is provided, just simply compare the two
     constraints with ALL solver time
    :return:
    """
    # previous outdated function, do not run. just to give an overview of how the constraints are compared
    def plot_condition_comparison(df, single_condition):
        # single_condition = 'distinct'
        def create_identifier(row, df_columns, single_condition):
            excluded_cols = ['average time',
                             'number of rows', 'percentage with any timeouts',
                             'avg nr of timeouts', 'full/holes ratio', single_condition]
            values = []
            for col in df_columns:
                if col not in excluded_cols:
                    values.append(str(row[col]))
            return '_'.join(values)

        df['config_id'] = df.apply(lambda row: create_identifier(row, df.columns, single_condition), axis=1)
        true_times = df[df[single_condition] == True]
        false_times = df[df[single_condition] == False]

        # merge dataframe so same conditions (except for single_condition) appears on the same row
        merged_times = pd.merge(true_times, false_times, on='config_id',
                                suffixes=('_' + single_condition, '_not_' + single_condition))
        # Create a new figure with 1 row and 2 columns of subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as per your requirement
        # Extract rows where 'generating full grid' is True and False respectively for distinct and pbeq
        full_true = merged_times[(merged_times['generating full grid_' + single_condition] == True) & (
                    merged_times['generating full grid_not_' + single_condition] == True)]
        full_false = merged_times[(merged_times['generating full grid_' + single_condition] == False) & (
                    merged_times['generating full grid_not_' + single_condition] == False)]

        # Subplot 1: Generating Full Grid
        axs[0].scatter(full_true['average time_' + single_condition], full_true['average time_not_' + single_condition],
                       alpha=0.5)
        axs[0].set_title('Generating Full Grid')
        axs[0].set_xlabel(f'Time using {single_condition}')
        axs[0].set_ylabel(f'Time using not {single_condition}')
        axs[0].grid(True)
        min_val_0 = min(min(full_true['average time_' + single_condition]),
                        min(full_true['average time_not_' + single_condition]))
        max_val_0 = max(max(full_true['average time_' + single_condition]),
                        max(full_true['average time_not_' + single_condition]))
        axs[0].plot([min_val_0, max_val_0], [min_val_0, max_val_0], color='red', linestyle='--', lw=2)

        # Subplot 2: Not Generating Full Grid
        axs[1].scatter(full_false['average time_' + single_condition],
                       full_false['average time_not_' + single_condition], alpha=0.5, color='green')
        axs[1].set_title('Generating Grid With Holes')
        axs[1].set_xlabel(f'Time using {single_condition}')
        axs[1].set_ylabel(f'Time using not {single_condition}')
        axs[1].grid(True)
        min_val_1 = min(min(full_false['average time_' + single_condition]),
                        min(full_false['average time_not_' + single_condition]))
        max_val_1 = max(max(full_false['average time_' + single_condition]),
                        max(full_false['average time_not_' + single_condition]))
        axs[1].plot([min_val_1, max_val_1], [min_val_1, max_val_1], color='red', linestyle='--', lw=2)

        # Adjust spacing between subplots
        plt.tight_layout(pad=4.0)

        # Display the plots
        plt.show()


if __name__ == '__main__':
    # dictionary of file paths to feed into `run_experiment`
    TIME_OUT = 5
    dct = {"curr_line_path": 'curr_line.txt',
           "classic_full_path": '../store-sudoku/classic_full_sudokus.txt',
           "argyle_full_path": '../store-sudoku/argyle_full_sudokus.txt',
           "classic_holes_path": '../store-sudoku/classic_holes_sudokus.txt',
           "argyle_holes_path": "../store-sudoku/argyle_holes_sudokus.txt"}
    # Left off with argyle-distinct-inorder-is_num-no_prefill-full_timeTotal

    hard_instances_file_dir = "hard_sudoku_instance-logFile/"
    alternative_solve_curr_line_path = "hard_sudoku_instance-logFile/curr_instance_line.txt"
    load_and_alternative_solve(hard_instances_file_dir, is_classic=True, num_iter=10,
                               currline_path=alternative_solve_curr_line_path, timeout=TIME_OUT)
    load_and_alternative_solve(hard_instances_file_dir, is_classic=False, num_iter=10,
                               currline_path=alternative_solve_curr_line_path, timeout=TIME_OUT)

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

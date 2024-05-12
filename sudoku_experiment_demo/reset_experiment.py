from jz3.utils.helpers import *
import jz3 as z3

def delete_experiment():
    dirs_clean = [
        './time-record',
        './time-record/particular_hard_instance_time_record',
        './time-record/whole_problem_time_records',
        './sudoku_database'
    ] #only clear the content, the files remain

    dirs_delete = [
        './problems_instances/particular_hard_instances_records/smt2_files',
        './problems_instances/particular_hard_instances_records/txt_files',
        './problems_instances/whole_problem_records/smt2_files',
        './problems_instances/whole_problem_records/txt_files'
    ] # delete everything in the directory
    clean_dir(dirs_clean)
    delete_dir(dirs_delete)

# Use the debugging function on your directory

if __name__ == '__main__':
    delete_experiment()
    # shuffle_and_remove_duplicates_files('./sudoku_database')

import numpy as np
import pandas as pd
import glob
from displayfunction import display
from pandas import ExcelWriter

# install openyxl
# install xlwt


# TODO: Adjust penalty weight
penalty_weight = 0  # seconds
excel_file_path = 'test.xlsx'


def useless_function():
    """
    :return: A dictionary mapping the file-name to it's corresponding average time
    """
    conditions = []
    ave_time = []
    dct = {}
    for file in glob.glob("time-record/*.txt"):
        with open(file, 'r') as f:
            a = f.read().split('\n')[:-1]  # strip the last whitespace
            # ['36.175875663757324,1', '36.49494504928589,1']
            a = np.average([float(ele.split(',')[0]) + int(ele.split(',')[1]) * penalty_weight for ele in a])
            print(a)
            conditions.append(f.name.replace('time-record/', '').replace('.txt', ''))
            ave_time.append(a)
            dct[f.name.replace('time-record/', '').replace('.txt', '')] = a


def create_table():
    """
    Creates a table containing the average time of each constraint
    :return: the table: 2D list
    """
    table = [[b1, b2, b3, b4, b5] for b1 in (True, False)
             for b2 in (True, False)
             for b3 in (True, False)
             for b4 in (True, False) if not (b2 and b4)
             for b5 in (True, False)]

    for ele in table:
        b1, b2, b3, b4, b5 = ele
        for suffix in ['full_time.txt', 'holes_time.txt']:
            s = ['argyle-', 'classic-'][b1] + ['PbEq-', 'distinct-'][b2] + ['inorder-', 'percol-'][b3] \
                + ['is_num-', 'is_bool-'][b4] + ['no_prefill-', 'prefill-'][b5] + suffix
            try:
                with open('time-record/' + s) as f:
                    lines = f.read().split('\n')[:-1]  # strip the last whitespace
            except FileNotFoundError:
                lines = []
            # ['36.175875663757324,1', '36.49494504928589,1']
            data = [line.split(',') for line in lines]
            runtimes = [float(datum[0]) for datum in data]
            timeouts = [int(datum[1]) for datum in data]
            average_time = np.average(runtimes)
            num_rows = len(data)
            # percentage of timeouts:
            timeout_perc = np.average([(0 if timeout == 0 else 1) for timeout in timeouts])
            # average timeout
            timeout_avg = np.average([timeout for timeout in timeouts if timeout != 0])

            yield [b1, b2, b3, b4, b5, suffix == 'full_time.txt',
                   average_time, num_rows, timeout_perc, timeout_avg]
    return table

if __name__ == '__main__':
    table = create_table()
    # export to excel
    df = pd.DataFrame(table, columns=['classic', 'distinct', 'per_col', 'no_num', 'prefill', 'generating full grid',
                                      'average time', 'number of rows', 'percentage with any timeouts', 'avg nr of timeouts'])
    time_rec = df['average time']
    solve_time = np.array(time_rec[::2])
    gen_time = np.array(time_rec[1::2])
    assert len(solve_time) == len(gen_time), "some full/holes sudokus are missing"
    df['full/holes ratio'] = [None if i % 2 == 0 else solve_time[i//2]/gen_time[i//2] for i in range(len(time_rec))]
    # print(df)
    with ExcelWriter(excel_file_path) as writer:
        df.to_excel(writer,sheet_name="Sheet1")
        # worksheet = writer.sheets["Sheet1"]
        # print(type(worksheet))
        # worksheet.cell()
        # worksheet.set_column(1, 1, 18)

    print("Finished")




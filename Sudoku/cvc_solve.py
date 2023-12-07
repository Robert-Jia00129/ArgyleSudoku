import subprocess
import time


if __name__ == '__main__':
    smt_log_file_path = "smt-logFiles/sudoku_smt_11_16_02_38_05_1700123885.859515.smt2"
    start_time = time.time()
    try:
        result = subprocess.run(["./cvc5-macOS-arm64", smt_log_file_path, "--lang", "smt2"],
                            capture_output=True, text=True, timeout=2)
    except subprocess.TimeoutExpired as exc:
        result = exc


    end_time = time.time()
    print(result.stdout)
    print(result.stderr)
    # combined_output = result.stdout + result.stderr # just to make sure
    # print(combined_output)
    # print(end_time-start_time,"cvc5 interrupted by timeout" in combined_output)
    print("FINISHED")
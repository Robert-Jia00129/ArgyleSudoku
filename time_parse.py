# this probably should be somewhere else
import sqlite3 as sql
from typing import Any, Generic, TypeVar, Sequence, cast

T = TypeVar("T", bound=dict)


class GenericDB(Generic[T]):
    """
    a wrapper for a sqlite db
    """

    def __init__(
        self,
        db_dir: str,
        table_name: str,
        schema: dict[str, str],
        id_column: bool = True,
        new: bool = True,
    ):
        self.conn = sql.connect(db_dir)
        self.cur = self.conn.cursor()
        self.table_name = table_name
        self.schema = schema
        self.columns = list(schema.keys())
        self.id_column = id_column

        if new:
            columns_def = ", ".join([f"{col} {dtype}" for col, dtype in schema.items()])
            if id_column:
                columns_def = f"ID INTEGER PRIMARY KEY AUTOINCREMENT, {columns_def}"
            try:
                self._exec(f"CREATE TABLE {table_name} ({columns_def});")
            except sql.OperationalError as e:
                raise Exception(f"Table creation failed: {e}\nMaybe it already exists?")

    def read_entries(self, filter_sql: str | None = None) -> list[T]:
        """Read entries with optional WHERE clause (e.g., 'time_taken > 3')"""
        where_clause = f"WHERE {filter_sql}" if filter_sql else ""
        cols = ", ".join(self.columns)
        query = f"SELECT {cols} FROM {self.table_name} {where_clause}"

        res = self.cur.execute(query)
        return [cast(T, dict(zip(self.columns, row))) for row in res.fetchall()]

    def add_entry(self, **kwargs: Any) -> None:
        """Add entry via keyword arguments: db.add_entry(time_taken=5.2, timeout_count=1)"""
        self.add_entries([kwargs])

    def add_entries(
        self, entries: Sequence[T | dict[str, Any] | Sequence[Any]]
    ) -> None:
        """Add multiple entries accepting dicts, TypedDicts, or sequences"""
        if not entries:
            return

        # Build parameterized query for safety
        cols = ", ".join(self.columns)
        placeholders = ", ".join(["?" for _ in self.columns])
        query = f"INSERT INTO {self.table_name} ({cols}) VALUES ({placeholders})"

        # Convert entries to standardized list of tuples
        entries_to_insert = []
        for entry in entries:
            if isinstance(entry, dict):
                # dict or TypedDict - extract values in schema order
                try:
                    values = tuple(entry[col] for col in self.columns)
                except KeyError as e:
                    raise ValueError(f"Missing column {e} in entry {entry}")
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                # Sequence (list/tuple) - must match column count
                if len(entry) != len(self.columns):
                    raise ValueError(
                        f"Sequence length {len(entry)} doesn't match column count {len(self.columns)}"
                    )
                values = tuple(entry)
            else:
                raise TypeError(f"Unsupported entry type: {type(entry)}")

            entries_to_insert.append(values)

        # Execute all inserts in a transaction
        try:
            self.cur.executemany(query, entries_to_insert)
            self.conn.commit()
        except sql.Error as e:
            self.conn.rollback()
            raise Exception(f"Insert failed: {e}")

    def _exec(self, lines: str | list[str], commit: bool = True) -> None:
        """Execute SQL line(s)"""
        if isinstance(lines, str):
            lines = [line.strip() + ";" for line in lines.split(";") if line.strip()]

        try:
            for line in lines:
                self.cur.execute(line)
            if commit:
                self.conn.commit()
        except sql.Error as e:
            self.conn.rollback()
            raise Exception(f"SQL execution failed: {e}")


def parse_whole_problem_time_records(
    file_directory: str, sql_db_directory: str, new: bool = True
) -> None:
    # read and parse from file to python obj
    with open(file_directory, "r") as f:
        # '...\n...'
        txt = f.read()
        # '[... , ... , ...]'
        times = txt.split("\n")
        times_parsed: list[tuple[float, int]] = []  # pyright: ignore
        for idx, t in enumerate(times):
            if "," not in t:
                continue  # possibly an empty line
            a, b = t.split(",")
            # '[(x1, y1), (x1, y1), (x1, y1), ...]'
            times_parsed.append((float(a), int(b)))

    # store to sql db
    # db = Time_db(sql_db_directory, "whole_problem_time_records", new)
    db = GenericDB(
        sql_db_directory,
        "whole_problem_time_records",
        {"time_taken": "FLOAT", "timeout_count": "INT"},
        id_column=True,
        new=True,
    )
    # db.add_entries(times_parsed)


def parse_particular_hard_instance(
    file_directory: str, sql_db_directory: str, new: bool = True
) -> None:
    # read and parse from file to python obj
    attempts_datas: list[dict] = []
    with open(file_directory, "r") as f:
        txt = f.read()
        times = txt.split("\n")
        for idx, t in enumerate(times):
            # parse the dict
            if t == "":
                continue
            d: dict = eval(t)
            attempts_datas.append(d)

    for idx, att in enumerate(attempts_datas):
        # problem data
        table_problem_data = GenericDB(
            sql_db_directory,
            f"instance{idx}_problem_data",
            {
                "grid": "STRING",
                "idx": "STRING",
                "try_Val": "INT",
                "assert_equals": "BOOL",
                "is_sat": "STRING",
            },
            id_column=True,
            new=True,
        )
        table_problem_data.add_entry(
            grid=att["problem"]["grid"],
            idx=str(att["problem"]["index"])[1:-1],  # [3, 6] -> '3, 6'
            try_Val=att["problem"]["try_Val"],
            assert_equals=att["problem"]["assert_equals"],
            is_sat=att["problem"]["is_sat"],
        )
        # attempts
        table_problem_data = GenericDB(
            sql_db_directory,
            f"instance{idx}_attempts_data",
            {
                "config": "STRING",
                "cvc5_time": "FLOAT",
                "cvc5_is_timeout": "BOOL",
                "cvc5_state": "STRING",
                "z3_time": "FLOAT",
                "z3_is_timeout": "BOOL",
                "z3_state": "STRING",
            },
            id_column=True,
            new=True,
        )
        for key in att.keys():
            if isinstance(key, str):
                continue
            table_problem_data.add_entry(
                config="".join(
                    map(lambda x: str(int(x)), key)
                ),  # (False, True, ...) -> '01...'
                cvc5_time=att[key]["cvc5"][0],
                cvc5_is_timeout=att[key]["cvc5"][1],
                cvc5_state=att[key]["cvc5"][2],
                z3_time=att[key]["z3"][0],
                z3_is_timeout=att[key]["z3"][1],
                z3_state=att[key]["z3"][2],
            )


if __name__ == "__main__":
    # parse_whole_problem_time_records(
    #     "./sudoku_experiment_demo/time-record/whole_problem_time_records/argyle-PbEq-inorder-is_num-prefill-full_time.txt",
    #     "./argyle-PbEq-inorder-is_num-prefill-full_time.db",
    #     # True,
    #     False,
    # )
    parse_particular_hard_instance(
        "./sudoku_experiment_demo/time-record/particular_hard_instance_time_record/argyle_time.txt",
        "./argyle_time.db",
    )

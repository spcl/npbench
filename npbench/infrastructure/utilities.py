# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import argparse
import numpy as np
import sqlite3
import timeit

from numbers import Number
from typing import Any, Dict, Union


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def time_to_ms(raw: float) -> int:
    return int(round(raw * 1000))


def relative_error(ref: Union[Number, np.ndarray], val: Union[Number, np.ndarray]) -> float:
    return np.linalg.norm(ref - val) / np.linalg.norm(ref)


# Taken from shttps://www.sqlitetutorial.net/sqlite-python/create-tables/
def create_connection(db_file) -> sqlite3.Connection:
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


# Taken from https://www.sqlitetutorial.net/sqlite-python/create-tables/
def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


def create_result(conn, query, result):
    """
    Create a new result into the results table
    :param conn:
    :param project:
    :return: project id
    """
    cur = conn.cursor()
    cur.execute(query, result)
    conn.commit()
    return cur.lastrowid


sql_create_results_table = """
CREATE TABLE IF NOT EXISTS results (
    id integer PRIMARY KEY,
    timestamp integer NOT NULL,
    benchmark text NOT NULL,
    kind text,
    domain text,
    dwarf text,
    preset text NOT NULL,
    mode text NOT NULL,
    framework text NOT NULL,
    version text NOT NULL,
    details text,
    validated integer,
    time real
);
"""

sql_insert_into_results_table = """
INSERT INTO results(
    timestamp, benchmark, kind, domain, dwarf, preset, mode,
    framework, version, details, validated, time
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

sql_create_lcounts_table = """
CREATE TABLE IF NOT EXISTS lcounts (
    id integer PRIMARY KEY,
    timestamp integer NOT NULL,
    benchmark text NOT NULL,
    kind text,
    domain text,
    dwarf text,
    mode text NOT NULL,
    framework text NOT NULL,
    version text NOT NULL,
    details text,
    count integer,
    npdiff integer
);
"""

sql_insert_into_lcounts_table = """
INSERT INTO lcounts(
    timestamp, benchmark, kind, domain, dwarf, mode,
    framework, version, details, count, npdiff
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

timeit_tmpl = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        {stmt}
    _t1 = _timer()
    return _t1 - _t0, {output}
"""


def benchmark(stmt, setup="pass", out_text="", repeat=1, context={}, output=None, verbose=True):

    ldict = {**context}
    raw_time_list = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
    raw_time = np.median(raw_time_list)
    ms_time = time_to_ms(raw_time)
    if verbose:
        print("{}: {}ms".format(out_text, ms_time))
    
    if output is not None:
        exec(setup, context)
        exec(stmt, context)
        res = context[output]
    else:
        res = None

    return res, raw_time_list


def validate(ref, val, framework="Unknown", rtol=1e-5, atol=1e-8, norm_error=1e-5):
    valid = True
    if not isinstance(ref, (tuple, list)):
        ref = [ref]
    if not isinstance(val, (tuple, list)):
        val = [val]
    # We do this check instead of strict=True in zip to give a more informative error message
    if len(ref) > len(val):
        print(f"{framework} did not return enough elements. Maybe you forgot a return statement?")
        valid = False
    for r, v in zip(ref, val):
        if f"{type(v).__module__}.{type(v).__name__}" == "torch.Tensor":
            v = v.cpu().numpy()

        if not np.allclose(r, v, rtol=rtol, atol=atol):
            try:
                import cupy
                if isinstance(v, cupy.ndarray):
                    relerror = relative_error(r, cupy.asnumpy(v))
                else:
                    relerror = relative_error(r, v)
            except Exception:
                relerror = relative_error(r, v)
            if relerror < norm_error:
                continue
            valid = False
            print("Relative error: {}".format(relerror))
            # return False
    if not valid:
        print("{} did not validate!".format(framework))
    return valid

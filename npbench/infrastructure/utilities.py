# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import argparse
import sqlite3

from typing import Union


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

"""
Results management module for Simple-FL
"""

import json
import time
import numpy as np
import pymysql
import sqlite3


def create_table(args, table_name):
    """
    Create database table for storing results
    
    Args:
        args: Arguments object
        table_name: Name of the table to create
    """
    # generate sql
    names = ""
    for k in args.as_dict().keys():
        if k == "recorder":
            for rk in args.recorder.keys():
                names += rk + " VARCHAR(255), "
            names += "recorder" + " TEXT, "
        else:
            names += k + " VARCHAR(255), "

    sql1 = (
        "CREATE TABLE IF NOT EXISTS {} (exp_id INT UNSIGNED AUTO_INCREMENT, ".format(
            table_name
        )
        + names.replace("TEXT", "LONGTEXT")
        + "submission_date DATETIME, PRIMARY KEY(exp_id));"
    )
    sql2 = (
        "CREATE TABLE IF NOT EXISTS {} (exp_id INTEGER PRIMARY KEY AUTOINCREMENT, ".format(
            table_name
        )
        + names
        + "submission_date DATETIME);"
    )
    # execute sql
    try:
        sql_execute(sql2, "local")
    except:
        pass
    sql_execute(sql1, "remote")


def sql_add_column():
    """Add column to existing table (utility function)"""
    table = "FL"
    sql = "ALTER TABLE {} ADD acc VARCHAR(255) AFTER ndcg10".format(table)
    sql = "ALTER TABLE {} ADD acc VARCHAR(255)".format(table)


def sql_insert(args, table_name):
    """
    Insert experiment results into database
    
    Args:
        args: Arguments object containing results
        table_name: Name of the table
    """
    # generate sql
    fields = " ( "
    values = " VALUES ("
    for k, v in args.items():
        if k == "recorder":
            for rk, rv in v.items():
                fields += str(rk) + ", "
                mean = np.mean(rv[-100:])
                std = np.std(rv[-100:])
                ms = {"mean": mean, "std": std}
                values += "'" + str(ms).replace("'", '"') + "'" + ", "
        if isinstance(v, dict):
            fields += str(k) + ", "
            values += "'" + str(v).replace("'", '"') + "'" + ", "
            json.dumps(v)
        elif isinstance(v, np.ndarray):
            fields += str(k) + ", "
            values += "'" + str(v).replace("'", '"') + "'" + ", "
        else:
            fields += str(k) + ", "
            values += "'" + str(v) + "'" + ", "
    fields += "submission_date)"
    values += "'" + time.strftime("%Y-%m-%d %H:%M:%S") + "'"
    sql = "INSERT INTO " + table_name + fields + values + ");"
    # execute sql
    try:
        sql_execute(sql, "local")
    except:
        pass
    sql_execute(sql, "remote")


def save_results(args, table_name):
    """
    Save experiment results to database
    
    Args:
        args: Arguments object containing results
        table_name: Name of the table
    """
    if "-" in table_name:
        table_name = table_name.replace("-", "_")
    create_table(args, table_name)
    sql_insert(args, table_name)


def sql_execute(sql, loc=None):
    """
    Execute SQL query
    
    Args:
        sql: SQL query string
        loc: Location ('remote' or 'local')
        
    Returns:
        Query results
    """
    if loc == "remote":
        con1 = pymysql.connect(
            host="10.72.74.136",
            port=13306,
            user="st",
            passwd="st2318822",
            db="results",
        )
        with con1:
            with con1.cursor(pymysql.cursors.DictCursor) as cur1:
                cur1.execute(sql)
                con1.commit()
                results = cur1.fetchall()
    elif loc == "local":

        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0]] = row[idx]
            return d

        con2 = sqlite3.connect("results.db")
        con2.row_factory = dict_factory
        cur2 = con2.cursor()
        with con2:
            cur2.execute(sql)
            con2.commit()
            results = cur2.fetchall()
            cur2.close()
    return results


def select(sql):
    """
    Execute SELECT query
    
    Args:
        sql: SQL query string
    """
    sql = "select * from centralized where dataset like 'ml-1m%'"
    sql_execute(sql, "remote")

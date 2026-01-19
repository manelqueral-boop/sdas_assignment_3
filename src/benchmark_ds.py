from config import (
    DB_PATH,
    BENCHMARK_FILE
)
import os
import json

import sqlite3

#TODO
def load_tables(spark_session, db_name):
    """
    Loads all tables from a SQLite database into a Spark session.

    Args:
        spark_session: Spark session to use for loading tables.
        db_name: Name of the SQLite database file to load tables from.
    """

    parent_dir = os.path.dirname(os.getcwd())

    path_to_tables = os.path.join(parent_dir,"db",db_name,"database_description")

    path_to_sql_file = os.path.join(parent_dir,"db",db_name)

    for table_name in os.listdir(path_to_tables):

        df = spark_session.read.format('jdbc').options(driver='org.sqlite.JDBC', dbtable=table_name[:-4],
                 url="jdbc:sqlite:"+os.path.join(path_to_sql_file,db_name+".sqlite")).load()



def load_query_info(query_id: int):

    #had to move query_workflow to src, so this finds it in db
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)

    query_data_file = os.path.join(parent_dir,DB_PATH, BENCHMARK_FILE)
    with open(query_data_file, 'r') as f:
        all_queries = json.load(f)

    query_info = None
    for query_entry in all_queries:
        if query_entry['question_id'] == query_id:
            query_info = query_entry
            break

    if query_info is None:
        raise ValueError(f"Query ID {query_id} not found")

    database_name = query_info['db_id']
    question = " ".join([
        query_info["question"],
        query_info["evidence"]
    ])
    golden_query = query_info["SQL"]

    return database_name, question, golden_query

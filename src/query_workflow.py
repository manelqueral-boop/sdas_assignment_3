import argparse
import os

import dotenv
import pandas as pd
import json

from spark_nl import (
    get_spark_session,
    get_spark_sql,
    get_spark_agent,
    run_nl_query,
    process_result,
    print_results,
    pretty_print_cot,
    run_sparksql_query,
    save_results
)
from benchmark_ds import (
    load_tables,
    load_query_info
)
from llm import get_llm
from evaluation import (
    translate_sqlite_to_spark,
    jaccard_index,
    result_to_obj,
    evaluate_spark_sql
)

def benchmark_query(query_id, provider):

    dotenv.load_dotenv()

    spark_session = get_spark_session()

    database_name, nl_query, golden_query = load_query_info(query_id)
    golden_query_spark = translate_sqlite_to_spark(golden_query)
    print(f"--- Benchmarking Query ID {query_id} on Database '{database_name}' ---")
    load_tables(spark_session, database_name)
    #exit()
    spark_sql = get_spark_sql()
    llm = get_llm(provider=provider)
    agent = get_spark_agent(spark_sql, llm=llm)
    run_nl_query(agent, nl_query, llm=llm)
    json_result = process_result()
    #with open("test.json", "r") as f:
    #    json_result = json.load(f)
    print_results(json_result)
    save_results(json_result)
    # pretty_print_cot(json_result)
    
    print(f"NL Query: \033[92m{nl_query}\033[0m")
    print(f"Golden Query (Spark SQL): \033[93m{golden_query_spark}\033[0m")

    if json_result["execution_status"] == "VALID":
        ground_truth_df = run_sparksql_query(spark_session, golden_query_spark)
        print("Ground Truth:")
        ground_truth_df.show()

        # Execution Accuracy
        inferred_result = run_sparksql_query(spark_session,json_result["sparksql_query"])#json_result["query_result"]
        print(type(inferred_result))
        #if isinstance(inferred_result,list):
        #    inferred_result = spark_session.createDataFrame(inferred_result,schema=["Result"])
        #    #exit()
        print("Type of ground truth: "+str(type(ground_truth_df)))
        print(type(inferred_result))
        ea = jaccard_index(ground_truth_df, inferred_result)
        print(f"Jaccard Index: {ea}")

        # Structural Accuracy        
        spark_sql_query = json_result.get("sparksql_query")
        if spark_sql_query:
            em_score = evaluate_spark_sql(golden_query_spark, spark_sql_query, spark_session)
            print(f"Spider Exact Match Score: {em_score}")

        if args.save is not None:
            filename = args.save
            output = {
                "execution_status": json_result["execution_status"],
                "jaccard": ea,
                "spider": em_score,
                "nl_querry": nl_query,
                "llm_querry": json_result["sparksql_query"],
                "gold_querry": golden_query_spark,
                "llm_requests": json_result["llm_requests"],
                "total_time": json_result["total_time"],
                "spark_time": json_result["spark_time"],
                "translation_time": json_result["translation_time"]
            }
            if not filename.endswith(".json"):
                filename = filename + ".json"
            output_path = os.path.join(os.getcwd(),filename)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=4)
    elif json_result["execution_status"] == "ERROR":
        if args.save is not None:
            filename = args.save
            output = {
                "execution_status": json_result["execution_status"],
                "jaccard": None,
                "spider": None,
                "nl_querry": nl_query,
                "llm_querry": json_result["sparksql_query"],
                "gold_querry": golden_query_spark,
                "llm_requests": None,
                "total_time": None,
                "spark_time": None,
                "translation_time": None,
            }
            if not filename.endswith(".json"):
                filename = filename + ".json"
            output_path = os.path.join(os.getcwd(),filename)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a specific query ID.")
    parser.add_argument("--id", type=int, default=1, help="Query ID to benchmark (default: 1)")
    parser.add_argument("--provider", type=str, default="google", help="LLM provider (default: google)")
    parser.add_argument("--save", type=str, default=None, help="Save results to a json file")

    args = parser.parse_args()
    
    benchmark_query(args.id, args.provider)

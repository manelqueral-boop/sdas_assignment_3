import argparse
import json
import os
import random
from os import mkdir

from config import (
    DB_PATH,
    BENCHMARK_FILE
)

def evaluation(selected):
    # Create folder structure for generated 10*n files
    os.system('mkdir eval')
    for id in selected:
        print(f"Start evaluation of {id}:")
        os.system(f'mkdir '+os.path.join("eval",f"evaluation_{id}"))

        for i in range(0,5):
            #os.system(f"touch "+os.path.join("eval",f"evaluation_{id}",f"{id}_{i}"))
            os.system(f'python3 query_workflow.py --id {id} --save '+os.path.join("eval",f"evaluation_{id}",f"{id}_{i}"))
    return 0

def start():
    parent_dir = os.getcwd()
    #ToDo Ugly solution should be changed
    path= os.path.join(parent_dir,'..', DB_PATH, BENCHMARK_FILE)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    item_with_highest_id = max(data, key=lambda x: x["question_id"], default=None)
    print(f"ID: {item_with_highest_id["question_id"]}")
    highest_id = item_with_highest_id["question_id"]
    if args.n > highest_id:
        print("to many questions selected")
        return 1
    selected = random.sample(range(0,highest_id), args.n)

    evaluation(selected)

    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated evaluation.")
    parser.add_argument("--n", type=int, default=20, help="amount of arbitrary choosen questions")

    args = parser.parse_args()
    start()
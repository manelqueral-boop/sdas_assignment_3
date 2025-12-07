DB_PATH = "db"
BENCHMARK_FILE = "dev.json"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT_SUFIX = "Use ` for the in-query strings. Don't limit the result size."

metrics = {
    "total_time": -1,
    "spark_exec_time": -1,
    "translation_time": -1,
    "sparksql_query": None,
    "answer": None
}
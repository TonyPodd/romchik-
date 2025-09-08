# заглушка под БД (SQLite/Postgres позже)
from typing import List, Dict, Any
logs: List[Dict[str, Any]] = []
def add_log(event: Dict[str, Any]): logs.append(event)

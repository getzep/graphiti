from datetime import datetime

from neo4j import time as neo4j_time


def parse_db_date(neo_date: neo4j_time.DateTime | None) -> datetime | None:
    return neo_date.to_native() if neo_date else None

import asyncio
from typing import Tuple
from datetime import datetime
import logging

from neo4j import AsyncGraphDatabase
from openai import OpenAI

from core.nodes import SemanticNode, EpisodicNode, Node
from core.edges import SemanticEdge, EpisodicEdge, Edge

logger = logging.getLogger(__name__)


class Graphiti:
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.database = "neo4j"

    def close(self):
        self.driver.close()
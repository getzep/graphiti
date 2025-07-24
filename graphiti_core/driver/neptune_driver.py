"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import boto3
import numpy as np
import datetime
from langchain_aws.graphs import NeptuneGraph, NeptuneAnalyticsGraph
from opensearchpy import helpers, OpenSearch, Urllib3HttpConnection, Urllib3AWSV4SignerAuth

from collections.abc import Coroutine
from typing import Any, List

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession
from graphiti_core.helpers import DEFAULT_DATABASE

logger = logging.getLogger(__name__)

class NeptuneDriverSession(GraphDriverSession):
    def __init__(self, graph: NeptuneGraph):  # type: ignore[reportUnknownArgumentType]
        self.graph = graph

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Neptune, but method must exist
        pass

    async def close(self):
        # No explicit close needed for Neptune, but method must exist
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        # Neptune does not support argument for Label Set, so it's converted into an array of queries
        if isinstance(query, list):
            for cypher, params in query:
                query = _sanitize_parameters(query, params)
                print(cypher)
                print(params)
                self.graph.query(str(cypher), params)  # type: ignore[reportUnknownArgumentType]
        else:
            params = dict(kwargs)
            query = _sanitize_parameters(query, params)

            self.graph.query(str(query), params)  # type: ignore[reportUnknownArgumentType]
        return None


def _sanitize_parameters(query, params):
    if isinstance(query, list):
        queries = []
        for q in query:
            queries.append(_sanitize_parameters(q, params))
        return queries
    else:
        for k, v in params.items():
            if isinstance(v, datetime.datetime):
                params[k] = v.isoformat()
            elif isinstance(v, list):   
                # Handle lists that might contain datetime objects
                for i, item in enumerate(v):
                    if isinstance(item, datetime.datetime):
                        v[i] = item.isoformat()
                        query = query.replace(f'${k}', f'datetime(${k})')
                    if isinstance(item, dict):
                        query = _sanitize_parameters(query, v[i])

                # If the list contains datetime objects, we need to wrap each element with datetime()
                if any(isinstance(item, str) and 'T' in item for item in v):
                    # Create a new list expression with datetime() wrapped around each element
                    datetime_list = '[' + ', '.join(f'datetime("{item}")' if isinstance(item, str) and 'T' in item else repr(item) for item in v) + ']'
                    query = query.replace(f'${k}', datetime_list)
        return query


aoss_indices = [
    {"index_name": "node_name_and_summary",
     "body": {
            "mappings": {
                   "properties": {
                    "uuid": {
                        "type": "text"
                    },
                    "name": {
                        "type": "text"
                    },
                    "summary": {
                        "type": "text"
                    },
                    "group_id": {
                        "type": "text"
                    }
                }
            }
        },
        "query": {
            "query": {
                "multi_match": {
                    "query": "",
                    "fields": ["name", "summary", "group_id"]
                }
            }   
        }
    },
    {"index_name": "community_name",
     "body": {
            "mappings": {
                   "properties": {
                    "uuid": {
                        "type": "text"
                    },
                    "name": {
                        "type": "text"
                    },
                    "group_id": {
                        "type": "text"
                    }
                }
            }
        },
        "query": {
            "query": {
                "multi_match": {
                    "query": "",
                    "fields": ["name", "group_id"]
                }
            }   
        }
     },
    {"index_name": "episode_content",
     "body": {
            "mappings": {
                   "properties": {
                    "uuid": {
                        "type": "text"
                    },
                    "content": {
                        "type": "text"
                    },
                    "source": {
                        "type": "text"
                    },
                    "source_description": {
                        "type": "text"
                    },
                    "group_id": {
                        "type": "text"
                    }
                }
            }
        },
        "query": {
            "query": {
                "multi_match": {
                    "query": "",
                    "fields": ["content", "source", "source_description", "group_id"]
                }
            }   
        }
     },
    {"index_name": "edge_name_and_fact",
     "body": {
            "mappings": {
                   "properties": {
                    "uuid": {
                        "type": "text"
                    },
                    "name": {
                        "type": "text"
                    },
                    "fact": {
                        "type": "text"
                    },
                    "group_id": {
                        "type": "text"
                    }
                }
            }
        },
        "query": {
            "query": {
                "multi_match": {
                    "query": "",
                    "fields": ["name", "fact", "group_id"]
                }
            }   
        }
     }
]

class NeptuneDriver(GraphDriver):
    provider: str = 'neptune'

    def __init__(
        self,
        host: str,
        aoss_host: str,
        port: int = 8182,
        aoss_port: int = 443
    ):
        if host:
            if host.startswith('neptune-db://'):
                # This is a Neptune Database Cluster
                endpoint = host.replace('neptune-db://', '')
                self.client = NeptuneGraph(endpoint, port)
                logger.debug('Creating Neptune Database session for %s', host)
            elif host.startswith('neptune-graph://'):
                # This is a Neptune Analytics Graph
                graphId = host.replace('neptune-graph://', '')
                self.client = NeptuneAnalyticsGraph(graphId)
                logger.debug('Creating Neptune Graph session for %s', host)
            else:
                raise ValueError(
                    'You must provide an endpoint to create a NeptuneDriver as either neptune-db://<endpoint> or neptune-graph://<graphid>'
                )
        else:
            raise ValueError('You must provide an endpoint to create a NeptuneDriver')
        
        if aoss_host:
            session  = boto3.Session()
            self.aoss_client = OpenSearch(
                hosts=[{
                    'host': aoss_host,
                    'port': 443
                }],
                http_auth=Urllib3AWSV4SignerAuth(session.get_credentials(), session.region_name, 'aoss'),
                use_ssl=True,
                verify_certs=True,
                connection_class=Urllib3HttpConnection,
                pool_maxsize = 20
            )        
        else:
            raise ValueError('You must provide an AOSS endpoint to create a NeptuneDriver')

    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> dict:
        params = dict(kwargs)
        if isinstance(cypher_query_, list):
            for q in cypher_query_:
                await self.execute_query(q, **params)
        else:
            cypher_query_ = _sanitize_parameters(cypher_query_, params)
            try:

                result = self.client.query(cypher_query_, params=params)
            except Exception as e:
                print(e)
                print(cypher_query_)
                print(params)
                logger.error('Error executing query: %s', e)
                raise e

            return result, None, None



    def session(self, database: str) -> GraphDriverSession:
        return NeptuneDriverSession(graph = self.client)

    async def close(self) -> None:
        return self.client.client.close()

    def delete_all_indexes(
        self, database_: str = DEFAULT_DATABASE
    ) -> Coroutine[Any, Any, Any]:
        return self.delete_all_indexes_impl()
    
    async def delete_all_indexes_impl(
        self
    ) -> Coroutine[Any, Any, Any]:
        # No matter what happens above, always return True
        return True
    
    async def create_aoss_indices(self):
        for index in aoss_indices:
            index_name = index['index_name']
            client = self.aoss_client
            if not client.indices.exists(index=index_name):
                client.indices.create(
                    index=index_name,
                    body=index['body']
                )
    
    async def delete_aoss_indices(self):
        for index in aoss_indices:
            index_name = index['index_name']
            client = self.aoss_client
            if not client.indices.exists(index=index_name):
                client.indices.delete(
                    index=index_name
                )
    
    def run_aoss_query(self, name:str, query_text:str): 
        for index in aoss_indices:
            if name.lower() == index['index_name']:                
                index['query']['query']['multi_match']['query'] = query_text
                query = {
                    "size": 10,
                    "query": index['query']
                }
                resp = self.aoss_client.search(
                    body = query['query'],
                    index = index['index_name']
                )
                return resp
    
    def save_to_aoss(self, name:str, data:list[dict]) -> bool: 
        for index in aoss_indices:
            if name.lower() == index['index_name']:
                to_index = []
                for d in data:
                    item = {"_index": name}
                    for p in index['body']['mappings']['properties'].keys():
                        item[p] = d[p]
                    to_index.append(item)                
                helpers.bulk(self.aoss_client, to_index)
        
        return True


    def calculate_cosine_similarity(self, vector1:List[float], vector2:List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors using NumPy.
        """
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)

        if norm_vector1 == 0 or norm_vector2 == 0:
            return 0  # Handle cases where one or both vectors are zero vectors

        return dot_product / (norm_vector1 * norm_vector2)


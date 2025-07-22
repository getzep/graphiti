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
import datetime
from langchain_aws.graphs import NeptuneGraph, NeptuneAnalyticsGraph
from opensearchpy import helpers, OpenSearch, Urllib3HttpConnection, Urllib3AWSV4SignerAuth

from collections.abc import Coroutine
from typing import Any

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession
from graphiti_core.helpers import DEFAULT_DATABASE

logger = logging.getLogger(__name__)


aoss_indices = [
    {"index_name": "node_name_and_summary",
     "body": {
            "mappings": {
                   "properties": {
                    "id": {
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
                    "id": {
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
                    "id": {
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
                    "id": {
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
        for k, v in params.items():
            if isinstance(v, datetime.datetime):
                params[k] = v.isoformat()
                cypher_query_ = cypher_query_.replace(f'${k}', f'datetime(${k})')
        result = self.client.query(cypher_query_, params=params)

        return result, None, None

    def session(self, database: str) -> GraphDriverSession:
        return self.client.session(database=database)  # type: ignore

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
            dimensions = 1536
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
    
    def run_aoss_query(self, name, query_text): 
        for index in aoss_indices:
            if name.lower() == index['index_name']:
                query = {
                    "size": 10,
                    "query": index['query']
                }
                resp = self.aoss_client.search(
                    body = query['query'],
                    index = index['index_name']
                )
                return resp




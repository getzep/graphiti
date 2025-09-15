# graph-service

Graph service is a fast api server implementing the [graphiti](https://github.com/getzep/graphiti) package.


## Running Instructions

1. Ensure you have Docker and Docker Compose installed on your system.

2. Add `zepai/graphiti:latest` to your service setup

3. Make sure to pass the following environment variables to the service

   ```
   OPENAI_API_KEY=your_openai_api_key
   NEO4J_USER=your_neo4j_user
   NEO4J_PASSWORD=your_neo4j_password
   NEO4J_PORT=your_neo4j_port
   ```

4. This service depends on having access to a neo4j instance, you may wish to add a neo4j image to your service setup as well. Or you may wish to use neo4j cloud or a desktop version if running this locally.

   An example of docker compose setup may look like this:

   ```yml
   version: '3.8'
   
   services:
     graph:
       image: zepai/graphiti:latest
       ports:
         - "8000:8000"
       depends_on:
         neo4j:
           condition: service_healthy
       restart: unless-stopped
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - NEO4J_URI="bolt://neo4j:${NEO4J_PORT}"
         - NEO4J_USER=neo4j
         - NEO4J_PASSWORD=${NEO4J_PASSWORD}
   
     neo4j:
       image: neo4j:5.22.0
       ports:
         - "7474:7474"
         - "${NEO4J_PORT}:${NEO4J_PORT}"
       volumes:
         - neo4j_data:/data
       environment:
         - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]
         interval: 30s
         timeout: 10s
         retries: 5
         start_period: 30s
   
   volumes:
     neo4j_data:
   ```

5. Once you start the service, it will be available at `http://localhost:8000` (or the port you have specified in the docker compose file).

6. You may access the swagger docs at `http://localhost:8000/docs`. You may also access redocs at `http://localhost:8000/redoc`.

7. You may also access the neo4j browser at `http://localhost:7474` (the port depends on the neo4j instance you are using).

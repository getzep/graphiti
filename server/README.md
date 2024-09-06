# graph-service

Graph service is a fast api server implementing the Graphiti package.

## Running Instructions

1. Ensure you have Docker and Docker Compose installed on your system.

2. Clone the repository and navigate to the `graph-service` directory.

3. Create a `.env` file in the `graph-service` directory with the following content:

   ```
   OPENAI_API_KEY=your_openai_api_key
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_neo4j_password
   NEO4J_PORT=7687
   ```

   Replace `your_openai_api_key` and `your_neo4j_password` with your actual OpenAI API key and desired Neo4j password.

4. Run the following command to start the services:

   ```
   docker-compose up --build
   ```

5. The graph service will be available at `http://localhost:8000`.

6. You may access the swagger docs at `http://localhost:8000/docs`.

7. You may also access the neo4j browser at `http://localhost:7474`.

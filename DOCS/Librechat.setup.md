  Complete Setup Guide: Graphiti MCP + LibreChat + Neo4j on Unraid

  Prerequisites

  - LibreChat running in Docker on Unraid
  - Neo4j Docker container running on Unraid
  - OpenAI API key (or other LLM provider)
  - Access to your Unraid Docker network

  ---
  Step 1: Prepare Graphiti MCP Configuration

  1.1 Create a directory on Unraid for Graphiti MCP

  mkdir -p /mnt/user/appdata/graphiti-mcp/config

  1.2 Create .env file

  Create /mnt/user/appdata/graphiti-mcp/.env with your settings:

  # Neo4j Connection - IMPORTANT: Use your existing Neo4j container details
  # If your Neo4j container is named "neo4j", use: bolt://neo4j:7687
  # Replace with your actual container name if different
  NEO4J_URI=bolt://YOUR_NEO4J_CONTAINER_NAME:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
  NEO4J_DATABASE=neo4j

  # OpenAI Configuration (Required)
  OPENAI_API_KEY=sk-your-openai-api-key-here

  # LLM Model (default: gpt-5-mini)
  MODEL_NAME=gpt-5-mini

  # Concurrency Control - adjust based on your OpenAI tier
  # Tier 1 (free): 1-2, Tier 2: 5-8, Tier 3: 10-15
  SEMAPHORE_LIMIT=10

  # Group ID for namespacing (optional)
  GRAPHITI_GROUP_ID=main

  # Disable telemetry (optional)
  GRAPHITI_TELEMETRY_ENABLED=false

  1.3 Create config file

  Create /mnt/user/appdata/graphiti-mcp/config/config.yaml:

  server:
    transport: "http"
    host: "0.0.0.0"
    port: 8000

  llm:
    provider: "openai"
    model: "gpt-5-mini"
    max_tokens: 4096

    providers:
      openai:
        api_key: ${OPENAI_API_KEY}
        api_url: ${OPENAI_API_URL:https://api.openai.com/v1}

  embedder:
    provider: "openai"
    model: "text-embedding-3-small"
    dimensions: 1536

    providers:
      openai:
        api_key: ${OPENAI_API_KEY}

  database:
    provider: "neo4j"

    providers:
      neo4j:
        uri: ${NEO4J_URI}
        username: ${NEO4J_USER}
        password: ${NEO4J_PASSWORD}
        database: ${NEO4J_DATABASE:neo4j}
        use_parallel_runtime: false

  graphiti:
    group_id: ${GRAPHITI_GROUP_ID:main}
    user_id: ${USER_ID:mcp_user}
    entity_types:
      - name: "Preference"
        description: "User preferences, choices, opinions, or selections"
      - name: "Requirement"
        description: "Specific needs, features, or functionality that must be fulfilled"
      - name: "Procedure"
        description: "Standard operating procedures and sequential instructions"
      - name: "Location"
        description: "Physical or virtual places where activities occur"
      - name: "Event"
        description: "Time-bound activities, occurrences, or experiences"
      - name: "Organization"
        description: "Companies, institutions, groups, or formal entities"
      - name: "Document"
        description: "Information content in various forms"
      - name: "Topic"
        description: "Subject of conversation, interest, or knowledge domain"
      - name: "Object"
        description: "Physical items, tools, devices, or possessions"

  ---
  Step 2: Deploy Graphiti MCP on Unraid

  You have two options for deploying on Unraid:

  Option A: Using Unraid Docker Template (Recommended)

  1. Go to Docker tab in Unraid
  2. Click Add Container
  3. Fill in the following settings:

  Basic Settings:
  - Name: graphiti-mcp
  - Repository: lvarming/graphiti-mcp:latest  # Custom build with your changes
  - Network Type: bridge (or custom: br0 if you have a custom network)

  Port Mappings:
  - Container Port: 8000 → Host Port: 8000

  Path Mappings:
  - Config Path:
    - Container Path: /app/mcp/config/config.yaml
    - Host Path: /mnt/user/appdata/graphiti-mcp/config/config.yaml
    - Access Mode: Read Only

  Environment Variables:
  Add each variable from your .env file:
  NEO4J_URI=bolt://YOUR_NEO4J_CONTAINER_NAME:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=your_password
  NEO4J_DATABASE=neo4j
  OPENAI_API_KEY=sk-your-key-here
  GRAPHITI_GROUP_ID=main
  SEMAPHORE_LIMIT=10
  CONFIG_PATH=/app/mcp/config/config.yaml
  PATH=/root/.local/bin:${PATH}

  Extra Parameters:
  --env-file=/mnt/user/appdata/graphiti-mcp/.env

  Network: Ensure this container is on the same Docker network as your Neo4j and LibreChat containers.

  Option B: Using Docker Compose

  Create /mnt/user/appdata/graphiti-mcp/docker-compose.yml:

  version: '3.8'

  services:
    graphiti-mcp:
      image: lvarming/graphiti-mcp:latest  # Custom build with your changes
      container_name: graphiti-mcp
      restart: unless-stopped
      env_file:
        - .env
      environment:
        - NEO4J_URI=${NEO4J_URI}
        - NEO4J_USER=${NEO4J_USER}
        - NEO4J_PASSWORD=${NEO4J_PASSWORD}
        - NEO4J_DATABASE=${NEO4J_DATABASE:-neo4j}
        - GRAPHITI_GROUP_ID=${GRAPHITI_GROUP_ID:-main}
        - SEMAPHORE_LIMIT=${SEMAPHORE_LIMIT:-10}
        - CONFIG_PATH=/app/mcp/config/config.yaml
        - PATH=/root/.local/bin:${PATH}
      volumes:
        - ./config/config.yaml:/app/mcp/config/config.yaml:ro
      ports:
        - "8000:8000"
      networks:
        - unraid_network  # Replace with your network name

  networks:
    unraid_network:
      external: true  # Use existing Unraid network

  Then run:
  cd /mnt/user/appdata/graphiti-mcp
  docker-compose up -d

  ---
  Step 3: Configure Docker Networking

  Find Your Neo4j Container Name

  docker ps | grep neo4j

  The container name will be something like neo4j or neo4j-community. Use this exact name in your NEO4J_URI.

  Ensure Same Network

  All three containers (Neo4j, Graphiti MCP, LibreChat) should be on the same Docker network.

  Check which network your Neo4j is on:
  docker inspect YOUR_NEO4J_CONTAINER_NAME | grep NetworkMode

  Connect Graphiti MCP to the same network:
  docker network connect NETWORK_NAME graphiti-mcp

  ---
  Step 4: Configure LibreChat

  4.1 Add Graphiti MCP to LibreChat's librechat.yaml

  Edit your LibreChat configuration file (usually /mnt/user/appdata/librechat/librechat.yaml):

  # ... existing LibreChat config ...

  # Add MCP server configuration
  mcpServers:
    graphiti-memory:
      url: "http://graphiti-mcp:8000/mcp/"
      # For multi-user support with user-specific graphs
      server_instructions: |
        You have access to a knowledge graph memory system through Graphiti.

        IMPORTANT USAGE GUIDELINES:
        1. Always search existing knowledge before adding new information
        2. Use entity type filters: Preference, Procedure, Requirement
        3. Store new information immediately using add_memory
        4. Follow discovered procedures and respect preferences

        Available tools:
        - add_episode: Store new conversations/information
        - search_nodes: Find entities and summaries
        - search_facts: Find relationships between entities
        - get_episodes: Retrieve recent conversations

      # Optional: Hide from chat menu (agent-only access)
      # chatMenu: false

      # Optional: User-specific group IDs for isolation
      # This requires configuring Graphiti to accept dynamic group_id
      # user_headers:
      #   X-User-ID: "{{LIBRECHAT_USER_ID}}"
      #   X-User-Email: "{{LIBRECHAT_USER_EMAIL}}"

  4.2 For Production: Use Streamable HTTP

  According to LibreChat docs, for multi-user deployments, ensure the transport is HTTP (which is already default for Graphiti MCP).

  4.3 Restart LibreChat

  docker restart YOUR_LIBRECHAT_CONTAINER_NAME

  ---
  Step 5: Test the Setup

  5.1 Verify Graphiti MCP is Running

  curl http://YOUR_UNRAID_IP:8000/health

  You should see a health status response.

  5.2 Test Neo4j Connection

  Check Graphiti MCP logs:
  docker logs graphiti-mcp

  Look for successful Neo4j connection messages.

  5.3 Test in LibreChat

  1. Open LibreChat in your browser
  2. Start a new chat
  3. In an agent configuration, you should see graphiti-memory available
  4. Try asking the agent to remember something:
  Please remember that I prefer dark mode for all interfaces
  5. Then later ask:
  What do you know about my preferences?

  ---
  Step 6: Advanced Configuration (Optional)

  Per-User Graph Isolation

  To give each LibreChat user their own knowledge graph, you need to:

  1. Modify Graphiti MCP to accept dynamic group_id from headers
  2. Update LibreChat config to send user info:

  mcpServers:
    graphiti-memory:
      url: "http://graphiti-mcp:8000/mcp/"
      user_headers:
        X-User-ID: "{{LIBRECHAT_USER_ID}}"
        X-User-Email: "{{LIBRECHAT_USER_EMAIL}}"

  This requires custom modification of Graphiti MCP to read the X-User-ID header and use it as the group_id.

  Using Different LLM Providers

  If you want to use Claude or other providers instead of OpenAI, update config.yaml:

  llm:
    provider: "anthropic"
    model: "claude-sonnet-4-5-latest"

    providers:
      anthropic:
        api_key: ${ANTHROPIC_API_KEY}

  And add ANTHROPIC_API_KEY to your .env file.

  ---
  Troubleshooting

  Graphiti MCP Can't Connect to Neo4j

  - Issue: Connection refused or timeout
  - Solution:
    - Verify Neo4j container name: docker ps | grep neo4j
    - Use exact container name in NEO4J_URI: bolt://container_name:7687
    - Ensure both containers are on same Docker network
    - Check Neo4j is listening on port 7687: docker exec NEO4J_CONTAINER netstat -tlnp | grep 7687

  LibreChat Can't See Graphiti Tools

  - Issue: Tools not appearing in agent builder
  - Solution:
    - Check Graphiti MCP is running: curl http://localhost:8000/health
    - Verify librechat.yaml syntax is correct
    - Restart LibreChat: docker restart librechat
    - Check LibreChat logs: docker logs librechat

  Rate Limit Errors

  - Issue: 429 errors from OpenAI
  - Solution: Lower SEMAPHORE_LIMIT in .env (try 5 or lower)

  Memory/Performance Issues

  - Issue: Slow responses or high memory usage
  - Solution:
    - Adjust Neo4j memory in your Neo4j container settings
    - Reduce SEMAPHORE_LIMIT to lower concurrent processing

  ---

⏺ Quick Start Summary

  Building Your Custom Docker Image:

  This setup uses a custom Docker image built from YOUR fork with YOUR changes.
  The image is automatically built by GitHub Actions and pushed to Docker Hub.

  Setup Steps:
  1. Fork the graphiti repository to your GitHub account
  2. Add Docker Hub credentials to your repository secrets:
     - Go to Settings → Secrets and variables → Actions
     - Add secret: DOCKERHUB_TOKEN (your Docker Hub access token)
  3. Push changes to trigger automatic build, or manually trigger from Actions tab
  4. Image will be available at: lvarming/graphiti-mcp:latest

  Key Points:

  1. Docker Image: Use lvarming/graphiti-mcp:latest (your custom build)
  2. Port: Expose 8000 for HTTP transport
  3. Neo4j Connection: Use bolt://YOUR_NEO4J_CONTAINER_NAME:7687 (container name, not localhost)
  4. Network: All 3 containers (Neo4j, Graphiti MCP, LibreChat) must be on same Docker network
  5. LibreChat Config: Add to librechat.yaml under mcpServers with URL: http://graphiti-mcp:8000/mcp/
  6. Required Env Vars: OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

  The setup gives LibreChat powerful knowledge graph memory capabilities, allowing it to remember user preferences, procedures, and facts across conversations!

  Let me know if you need help with any specific step or run into issues during setup.
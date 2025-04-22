#!/bin/bash
# Graphiti CLI test script
# This script tests all the graphiti CLI commands with safeguards to prevent accidental data loss

set -e  # Exit on error

# Configuration
GRAPHITI_DIR="/Users/dmieloch/mcp-servers/graphiti"
TEST_JSON_FILE="${GRAPHITI_DIR}/test_data.json"
TEST_NAME="CLI Test Script"
TEST_PREFIX="TEST_SAFE_TO_DELETE"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
  echo -e "\n${BLUE}========== $1 ==========${NC}\n"
}

print_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
  echo -e "${RED}❌ $1${NC}"
}

wait_for_user() {
  echo -e "\n${YELLOW}Press ENTER to continue...${NC}"
  read -r
}

# Create test directory if it doesn't exist
mkdir -p "${GRAPHITI_DIR}/test"

# Create a test JSON file
cat > "$TEST_JSON_FILE" << EOL
{
  "test_id": "${TEST_PREFIX}_$(date +%s)",
  "name": "${TEST_PREFIX} Test Object",
  "description": "This is a test object created by the CLI test script",
  "test_date": "$(date -Iseconds)",
  "properties": {
    "safe_to_delete": true,
    "purpose": "automated testing"
  }
}
EOL

# Start testing
print_header "GRAPHITI CLI TEST SCRIPT"
echo "This script will test all graphiti CLI commands."
echo "Test data will be marked with prefix: ${TEST_PREFIX}"
wait_for_user

# Test connection
print_header "Testing Neo4j Connection"
cd "$GRAPHITI_DIR" && uv run cli/main.py check-connection
wait_for_user

# Test add-json
print_header "Testing add-json"
cd "$GRAPHITI_DIR" && uv run cli/main.py add-json --json-file "$TEST_JSON_FILE" --name "${TEST_PREFIX} JSON File Test"
print_success "Added JSON from file"
wait_for_user

# Test add-json-string
print_header "Testing add-json-string"
cd "$GRAPHITI_DIR" && uv run cli/main.py add-json-string --json-data '{"test_id":"'${TEST_PREFIX}'_string","content":"This is a test JSON string"}' --name "${TEST_PREFIX} JSON String Test"
print_success "Added JSON from string"
wait_for_user

# Test search-nodes
print_header "Testing search-nodes"
cd "$GRAPHITI_DIR" && uv run cli/main.py search-nodes --query "${TEST_PREFIX}"
print_success "Searched for nodes"
echo "Please verify that you see the test nodes in the results."
echo "Copy a node UUID from the results for the next tests:"
read -p "Node UUID: " TEST_NODE_UUID
wait_for_user

if [ -z "$TEST_NODE_UUID" ]; then
  print_warning "No UUID provided, some tests will be skipped"
else
  # Test search-nodes with center
  print_header "Testing search-nodes with center node"
  cd "$GRAPHITI_DIR" && uv run cli/main.py search-nodes --query "${TEST_PREFIX}" --center "$TEST_NODE_UUID" --max 3
  print_success "Searched for nodes with center node"
  wait_for_user

  # Test search-facts with center
  print_header "Testing search-facts with center node" 
  cd "$GRAPHITI_DIR" && uv run cli/main.py search-facts --query "test" --center "$TEST_NODE_UUID" --max 3
  print_success "Searched for facts with center node"
  wait_for_user
fi

# Test search-facts
print_header "Testing search-facts"
cd "$GRAPHITI_DIR" && uv run cli/main.py search-facts --query "${TEST_PREFIX}"
print_success "Searched for facts"
echo "Please verify that you see relationships related to the test nodes."
echo "Copy a fact/edge UUID from the results for the next tests:"
read -p "Edge UUID: " TEST_EDGE_UUID
wait_for_user

if [ -z "$TEST_EDGE_UUID" ]; then
  print_warning "No Edge UUID provided, some tests will be skipped"
else
  # Test get-entity-edge
  print_header "Testing get-entity-edge"
  cd "$GRAPHITI_DIR" && uv run cli/main.py get-entity-edge --uuid "$TEST_EDGE_UUID"
  print_success "Retrieved entity edge details"
  wait_for_user
fi

# Test get-episodes
print_header "Testing get-episodes"
cd "$GRAPHITI_DIR" && uv run cli/main.py get-episodes
print_success "Retrieved episodes"
echo "Please verify that you see the test episodes in the results."
echo "Copy an episode UUID from the results for the next tests:"
read -p "Episode UUID: " TEST_EPISODE_UUID
wait_for_user

# Testing deletion commands - with special safeguards
if [ -n "$TEST_EDGE_UUID" ]; then
  print_header "Testing delete-entity-edge PREVIEW"
  print_warning "This will only PREVIEW deletion, not actually delete"
  cd "$GRAPHITI_DIR" && uv run cli/main.py get-entity-edge --uuid "$TEST_EDGE_UUID"
  wait_for_user

  echo -e "\n${YELLOW}Would you like to test actual edge deletion? This will remove the edge from the database. [y/N]${NC}"
  read -r confirm_edge_deletion
  if [[ $confirm_edge_deletion =~ ^[Yy]$ ]]; then
    print_header "Testing delete-entity-edge"
    cd "$GRAPHITI_DIR" && uv run cli/main.py delete-entity-edge --uuid "$TEST_EDGE_UUID" --confirm
    print_success "Tested edge deletion"
  else
    print_warning "Skipping actual edge deletion"
  fi
  wait_for_user
fi

if [ -n "$TEST_EPISODE_UUID" ]; then
  print_header "Testing delete-episode PREVIEW"
  print_warning "This will only PREVIEW deletion, not actually delete"
  cd "$GRAPHITI_DIR" && uv run cli/main.py delete-episode --uuid "$TEST_EPISODE_UUID"
  wait_for_user

  echo -e "\n${YELLOW}Would you like to test actual episode deletion? This will remove the episode from the database. [y/N]${NC}"
  read -r confirm_episode_deletion
  if [[ $confirm_episode_deletion =~ ^[Yy]$ ]]; then
    print_header "Testing delete-episode"
    cd "$GRAPHITI_DIR" && uv run cli/main.py delete-episode --uuid "$TEST_EPISODE_UUID" --confirm
    print_success "Tested episode deletion"
  else
    print_warning "Skipping actual episode deletion"
  fi
  wait_for_user
fi

# Testing clear-graph with extreme caution
print_header "Testing clear-graph interface"
print_warning "This will only TEST the interface, not clear the database"
echo "The test will stop before actual deletion"
echo -e "\n${YELLOW}Would you like to test the clear-graph interface? This WON'T delete data but shows the prompts. [y/N]${NC}"
read -r confirm_clear_test
if [[ $confirm_clear_test =~ ^[Yy]$ ]]; then
  # Create a temporary testing script that will exit before actual deletion
  TEMP_SCRIPT=$(mktemp)
  cat > "$TEMP_SCRIPT" << EOL
#!/bin/bash
echo "Testing clear-graph interface..."
echo "This is a simulation that will NOT actually clear the database."
cd "$GRAPHITI_DIR" 
# Show help for clear-graph to see parameters
uv run cli/main.py clear-graph --help
# Start the clear command but pipe a wrong confirmation code
echo "WRONG-CODE" | uv run cli/main.py clear-graph --confirm --force || true
echo "Test complete - database was not cleared."
EOL
  chmod +x "$TEMP_SCRIPT"
  "$TEMP_SCRIPT"
  rm "$TEMP_SCRIPT"
  print_success "Tested clear-graph interface without actual deletion"
else
  print_warning "Skipping clear-graph interface test"
fi

# Clean up
print_header "Cleaning up test data"
if [ -f "$TEST_JSON_FILE" ]; then
  rm "$TEST_JSON_FILE"
  print_success "Removed test JSON file"
fi

print_header "TEST SUMMARY"
echo "All tests have been completed."
echo "Please manually verify that:"
echo "1. The tests ran without errors"
echo "2. You could see the test data in search results"
echo "3. The deletion safeguards worked as expected"
echo ""
echo "To delete any remaining test data, use search to find items with prefix: ${TEST_PREFIX}"
echo "Then use the appropriate deletion commands to remove them." 
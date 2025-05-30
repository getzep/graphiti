# Browser Agent Testing Plan

This document outlines a comprehensive testing plan for the Graphiti Browser Agent feature. Each deliverable is accompanied by specific tests to verify functionality and identify any issues.

## 1. Browser Extension Structure

### Deliverable: Complete extension file structure
**Verification Tests:**
- [ ] Run `ls -la browser_agent/` and verify the following files exist:
  - manifest.json
  - popup.html
  - popup.css
  - popup.js
  - content.js
  - background.js
  - README.md
  - icons/ directory
- [ ] Validate manifest.json with Chrome extension validator:
  ```bash
  # Install extension validator if needed
  npm install -g chrome-extension-validator
  # Validate the manifest
  chrome-extension-validator browser_agent/
  ```

## 2. Browser Extension UI

### Deliverable: Functional popup UI with all required components
**Verification Tests:**
- [ ] Load the extension in Chrome developer mode:
  1. Navigate to `chrome://extensions/`
  2. Enable "Developer mode"
  3. Click "Load unpacked" and select the `browser_agent` directory
- [ ] Verify the extension icon appears in the toolbar
- [ ] Click the extension icon and verify the popup appears with:
  - Connection status indicator
  - Data extraction buttons
  - Results section
  - Category management section
  - Settings section
- [ ] Verify all UI elements are properly styled according to popup.css

## 3. Data Extraction Functionality

### Deliverable: Working data extraction from web pages
**Verification Tests:**
- [ ] Test automatic extraction:
  1. Navigate to a test website (e.g., `https://example.com`)
  2. Open the extension popup
  3. Click "Extract Data"
  4. Verify data appears in the extraction results section
  5. Check that the data includes URL, title, and content
- [ ] Test element selection:
  1. Navigate to a test website
  2. Open the extension popup
  3. Click "Select Elements"
  4. Verify the selection overlay appears
  5. Select several elements on the page
  6. Click "Finish Selection"
  7. Reopen the popup and extract data
  8. Verify only the selected elements are included in the results

## 4. Server API Implementation

### Deliverable: Functional server-side API endpoints
**Verification Tests:**
- [ ] Verify API module structure:
  ```bash
  # Check that the API file exists
  ls -la server/graph_service/routers/browser_agent.py
  # Check that it's imported in main.py
  grep -n "browser_agent" server/graph_service/main.py
  ```
- [ ] Test API endpoints using the test script:
  ```bash
  # Start the server in one terminal
  cd server
  python -m graph_service.main
  
  # In another terminal, run the test script
  cd browser_agent
  python test_api.py --url http://localhost:8000
  ```
- [ ] Verify all three endpoints return successful responses:
  - `/browser_agent/health` should return status "ok"
  - `/browser_agent/categorize` should return a list of categories
  - `/browser_agent/ingest` should return a success message with an episode ID

## 5. AI Categorization

### Deliverable: Working AI-powered categorization of web content
**Verification Tests:**
- [ ] Test categorization API directly:
  ```bash
  curl -X POST "http://localhost:8000/browser_agent/categorize" \
    -H "Content-Type: application/json" \
    -d '{"content": {"url": "https://example.com", "title": "Example Domain", "content": {"text": "This domain is for use in illustrative examples in documents."}}}'
  ```
- [ ] Verify the response contains a list of relevant categories
- [ ] Test categorization through the extension:
  1. Navigate to a content-rich website
  2. Extract data using the extension
  3. Verify AI categories appear in the categories section
  4. Test with different types of content to ensure varied categorization

## 6. Data Saving to Knowledge Graph

### Deliverable: Ability to save web data to Graphiti knowledge graph
**Verification Tests:**
- [ ] Test data ingestion API directly:
  ```bash
  curl -X POST "http://localhost:8000/browser_agent/ingest" \
    -H "Content-Type: application/json" \
    -d '{"content": {"url": "https://example.com", "title": "Example Domain", "content": {"text": "Example content"}}, "categories": ["Test", "Example"], "source": {"url": "https://example.com", "title": "Example Domain", "timestamp": "2025-05-21T12:00:00Z"}}'
  ```
- [ ] Verify the response contains a success message and episode ID
- [ ] Verify the data was added to the knowledge graph:
  ```bash
  # Query the knowledge graph for the new episode
  curl "http://localhost:8000/retrieve/episodes?query=Example%20Domain"
  ```
- [ ] Test saving through the extension:
  1. Extract data from a web page
  2. Add some custom categories
  3. Click "Save to Knowledge Graph"
  4. Verify success message appears
  5. Query the knowledge graph to confirm the data was saved

## 7. Cross-Origin Communication

### Deliverable: Working CORS support for browser extension
**Verification Tests:**
- [ ] Verify CORS middleware is properly configured:
  ```bash
  grep -n "CORSMiddleware" server/graph_service/main.py
  ```
- [ ] Test cross-origin requests:
  ```bash
  # Run this in the browser console while on a different domain
  fetch('http://localhost:8000/browser_agent/health', {
    method: 'GET',
    headers: {'Content-Type': 'application/json'}
  }).then(r => r.json()).then(console.log)
  ```
- [ ] Verify the request succeeds without CORS errors

## 8. Extension Settings

### Deliverable: Working settings management in the extension
**Verification Tests:**
- [ ] Test saving settings:
  1. Open the extension popup
  2. Enter a test API endpoint and key in the settings section
  3. Click "Save Settings"
  4. Close and reopen the popup
  5. Verify the settings are still populated
- [ ] Test connection status:
  1. Enter a valid API endpoint (e.g., `http://localhost:8000`)
  2. Save settings
  3. Verify the connection status changes to "Connected"
  4. Enter an invalid endpoint
  5. Save settings
  6. Verify the connection status changes to "Disconnected"

## 9. Documentation

### Deliverable: Complete and accurate documentation
**Verification Tests:**
- [ ] Verify README files exist and contain comprehensive information:
  ```bash
  ls -la browser_agent/README.md browser_agent/FEATURE_README.md
  ```
- [ ] Check documentation for:
  - Installation instructions
  - Usage instructions
  - API documentation
  - Testing procedures
  - Architecture overview
- [ ] Verify code comments:
  ```bash
  # Check for comments in key files
  grep -n "//" browser_agent/content.js | wc -l
  grep -n "#" server/graph_service/routers/browser_agent.py | wc -l
  ```

## 10. End-to-End Testing

### Deliverable: Fully functional browser agent system
**Verification Tests:**
- [ ] Perform a complete end-to-end test:
  1. Start the Graphiti server
  2. Load the browser extension
  3. Configure the extension with the server URL
  4. Navigate to a content-rich website
  5. Extract data using the extension
  6. Verify AI categories appear
  7. Add a custom category
  8. Save the data to the knowledge graph
  9. Query the knowledge graph to verify the data was saved with the correct categories
- [ ] Test with different types of websites:
  - News articles
  - Blog posts
  - Product pages
  - Documentation pages

## Test Reporting

For each test, document:
1. Test date and time
2. Test environment (browser version, OS)
3. Test result (Pass/Fail)
4. Any errors or issues encountered
5. Screenshots of the UI where applicable

## Continuous Integration Tests

Add the following tests to CI pipeline if available:

```yaml
browser-agent-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-asyncio
    - name: Test browser agent API
      run: |
        cd server
        pytest -xvs tests/test_browser_agent.py
    - name: Validate extension structure
      run: |
        npm install -g chrome-extension-validator
        chrome-extension-validator browser_agent/
```

## Error Recovery Plan

For each potential failure point, document:
1. Possible causes
2. Diagnostic steps
3. Recovery actions

Example:
- **Failure**: Data extraction returns empty results
  - **Causes**: Content script not injected, selectors not matching page structure
  - **Diagnosis**: Check browser console for errors, verify content script is loaded
  - **Recovery**: Update content selectors, check for page structure changes
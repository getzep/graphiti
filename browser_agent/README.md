# Graphiti Browser Agent

A browser extension that allows users to extract and categorize web page data and add it to the Graphiti knowledge graph.

## Features

- Extract data from web pages automatically or by selecting specific elements
- AI-powered categorization of extracted data
- Customizable semantic categorization
- Save web page data to the Graphiti knowledge graph
- User-friendly interface

## Installation

### Development Mode

1. Clone the repository
2. Navigate to `chrome://extensions/` in your Chrome browser
3. Enable "Developer mode" in the top right corner
4. Click "Load unpacked" and select the `browser_agent` directory

### Production Mode

1. Package the extension:
   ```
   zip -r graphiti-browser-agent.zip browser_agent/*
   ```
2. Upload to the Chrome Web Store (requires developer account)

## Usage

1. Click the Graphiti Browser Agent icon in your browser toolbar
2. Configure the Graphiti API endpoint in the settings
3. Use one of the extraction methods:
   - Click "Extract Data" to automatically extract content from the current page
   - Click "Select Elements" to manually select specific elements on the page
4. Review the extracted data and AI-suggested categories
5. Add or remove categories as needed
6. Click "Save to Knowledge Graph" to add the data to Graphiti

## Server Configuration

The browser agent requires a running Graphiti server with the browser agent API endpoints enabled. Make sure the server is properly configured and accessible from the browser.

### API Endpoints

- `/browser_agent/health` - Health check endpoint
- `/browser_agent/categorize` - AI categorization endpoint
- `/browser_agent/ingest` - Data ingestion endpoint

## Development

### Project Structure

- `manifest.json` - Extension configuration
- `popup.html` - Extension popup UI
- `popup.css` - Styles for the popup
- `popup.js` - JavaScript for the popup functionality
- `content.js` - Content script that interacts with web pages
- `background.js` - Background script for extension functionality

### Building and Testing

1. Make changes to the code
2. Reload the extension in Chrome by clicking the refresh icon on the extensions page
3. Test the functionality on various web pages

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
# Graphiti Browser Agent Feature

## Overview

The Graphiti Browser Agent is a new feature that extends Graphiti's knowledge graph capabilities to web browsers. It allows users to extract, categorize, and save web page data directly to their Graphiti knowledge graph.

## Key Features

1. **Web Data Extraction**: Extract structured and unstructured data from any web page
   - Automatic extraction of main content, metadata, and structured data
   - Manual selection of specific elements on the page

2. **AI-Powered Categorization**: Automatically categorize extracted data
   - Uses Graphiti's LLM capabilities to suggest relevant categories
   - Identifies entities, topics, and themes in the content

3. **Customizable Ontologies**: Allow users to customize categorization
   - Accept or modify AI-suggested categories
   - Add custom categories based on user knowledge

4. **Seamless Integration**: Direct integration with Graphiti knowledge graph
   - Save web data as episodes in the knowledge graph
   - Preserve metadata and source information
   - Maintain semantic relationships between extracted entities

## Components

### Browser Extension

- **User Interface**: Clean, intuitive popup interface for data extraction and categorization
- **Content Scripts**: JavaScript that interacts with web pages to extract data
- **Background Scripts**: Manages extension state and communication

### Server API

- **Categorization Endpoint**: AI-powered analysis of web content
- **Ingestion Endpoint**: Adds web data to the knowledge graph
- **Health Check**: Ensures connectivity between extension and server

## Use Cases

1. **Research**: Collect and organize information from multiple web sources
2. **Content Creation**: Build a knowledge base of reference materials
3. **Learning**: Save and categorize educational content
4. **Project Management**: Gather and organize project-related information
5. **Personal Knowledge Management**: Build a personalized knowledge graph

## Implementation Details

The feature consists of two main parts:

1. **Browser Extension**: A Chrome extension that users can install to interact with web pages
2. **Server API**: New endpoints added to the Graphiti server to handle browser agent requests

### Data Flow

1. User browses to a web page and activates the extension
2. User selects content to extract or uses automatic extraction
3. Extension sends data to the server for AI categorization
4. User reviews and customizes categories
5. Data is saved to the Graphiti knowledge graph
6. Information becomes available for querying and retrieval

## Getting Started

1. Install the browser extension
2. Configure the extension with your Graphiti server URL
3. Browse to a web page and click the extension icon
4. Extract data and customize categories
5. Save to your knowledge graph

## Future Enhancements

- Support for additional browsers (Firefox, Safari, Edge)
- Enhanced selection tools for more precise data extraction
- Integration with existing knowledge graph for contextual categorization
- Batch processing of multiple pages
- Scheduled crawling of specified websites
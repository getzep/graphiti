// Global variables
let isSelectionMode = false;
let selectedElements = [];

// Listen for messages from the popup
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "extractPageData") {
    const data = extractPageData();
    sendResponse({data: data});
  } else if (request.action === "enableElementSelection") {
    enableElementSelection();
    sendResponse({success: true});
  } else if (request.action === "testConnection") {
    // Simple test connection handler
    sendResponse({success: true, message: "Connection successful"});
  }
  return true; // Keep the message channel open for async responses
});

// Extract data from the current page
function extractPageData() {
  // Basic metadata
  const metadata = {
    url: window.location.href,
    title: document.title,
    timestamp: new Date().toISOString(),
    domain: window.location.hostname
  };
  
  // Extract main content
  const mainContent = extractMainContent();
  
  // Extract structured data if available
  const structuredData = extractStructuredData();
  
  // Extract meta tags
  const metaTags = extractMetaTags();
  
  // Combine all data
  return {
    ...metadata,
    content: mainContent,
    structuredData: structuredData,
    metaTags: metaTags
  };
}

// Extract the main content from the page
function extractMainContent() {
  // If elements were specifically selected, use those
  if (selectedElements.length > 0) {
    return selectedElements.map(el => {
      return {
        text: el.innerText.trim(),
        html: el.outerHTML,
        tag: el.tagName.toLowerCase(),
        classes: el.className
      };
    });
  }
  
  // Otherwise, try to identify the main content
  const mainContentSelectors = [
    'article',
    'main',
    '.content',
    '#content',
    '.main',
    '#main',
    '.article',
    '.post',
    '.entry'
  ];
  
  let mainContent = null;
  
  // Try each selector until we find content
  for (const selector of mainContentSelectors) {
    const element = document.querySelector(selector);
    if (element && element.innerText.trim().length > 100) {
      mainContent = {
        text: element.innerText.trim(),
        html: element.outerHTML,
        tag: element.tagName.toLowerCase(),
        classes: element.className
      };
      break;
    }
  }
  
  // If no main content found, use the body
  if (!mainContent) {
    const paragraphs = Array.from(document.querySelectorAll('p')).filter(p => 
      p.innerText.trim().length > 50 && 
      !p.closest('header') && 
      !p.closest('footer') && 
      !p.closest('nav')
    );
    
    if (paragraphs.length > 0) {
      mainContent = paragraphs.map(p => ({
        text: p.innerText.trim(),
        html: p.outerHTML,
        tag: 'p',
        classes: p.className
      }));
    } else {
      // Last resort: use body text
      mainContent = {
        text: document.body.innerText.trim(),
        html: document.body.outerHTML,
        tag: 'body',
        classes: document.body.className
      };
    }
  }
  
  return mainContent;
}

// Extract structured data (JSON-LD, microdata, etc.)
function extractStructuredData() {
  const structuredData = [];
  
  // Extract JSON-LD
  const jsonLdScripts = document.querySelectorAll('script[type="application/ld+json"]');
  jsonLdScripts.forEach(script => {
    try {
      const data = JSON.parse(script.textContent);
      structuredData.push({
        type: 'json-ld',
        data: data
      });
    } catch (e) {
      console.error('Error parsing JSON-LD:', e);
    }
  });
  
  // Extract microdata (simplified)
  const itemscopes = document.querySelectorAll('[itemscope]');
  itemscopes.forEach(item => {
    const itemtype = item.getAttribute('itemtype');
    const properties = {};
    
    item.querySelectorAll('[itemprop]').forEach(prop => {
      const name = prop.getAttribute('itemprop');
      let value;
      
      if (prop.tagName === 'META') {
        value = prop.getAttribute('content');
      } else if (prop.tagName === 'IMG') {
        value = prop.getAttribute('src');
      } else if (prop.tagName === 'A') {
        value = prop.getAttribute('href');
      } else if (prop.tagName === 'TIME') {
        value = prop.getAttribute('datetime') || prop.textContent.trim();
      } else {
        value = prop.textContent.trim();
      }
      
      properties[name] = value;
    });
    
    structuredData.push({
      type: 'microdata',
      itemtype: itemtype,
      properties: properties
    });
  });
  
  return structuredData;
}

// Extract meta tags
function extractMetaTags() {
  const metaTags = {};
  
  // Extract standard meta tags
  document.querySelectorAll('meta').forEach(meta => {
    const name = meta.getAttribute('name') || meta.getAttribute('property');
    const content = meta.getAttribute('content');
    
    if (name && content) {
      metaTags[name] = content;
    }
  });
  
  return metaTags;
}

// Enable element selection mode
function enableElementSelection() {
  if (isSelectionMode) return;
  
  isSelectionMode = true;
  selectedElements = [];
  
  // Create overlay and instructions
  const overlay = document.createElement('div');
  overlay.style.position = 'fixed';
  overlay.style.top = '0';
  overlay.style.left = '0';
  overlay.style.width = '100%';
  overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
  overlay.style.color = 'white';
  overlay.style.padding = '20px';
  overlay.style.zIndex = '9999';
  overlay.style.textAlign = 'center';
  overlay.innerHTML = `
    <h2>Select elements to extract</h2>
    <p>Click on elements to select them. Press ESC when done.</p>
    <div id="selected-count">Selected: 0 elements</div>
    <button id="finish-selection" style="padding: 8px 16px; margin-top: 10px; background-color: #4a6cf7; color: white; border: none; border-radius: 4px; cursor: pointer;">Finish Selection</button>
  `;
  document.body.appendChild(overlay);
  
  // Add hover effect style
  const style = document.createElement('style');
  style.textContent = `
    .graphiti-hover-highlight {
      outline: 2px dashed #4a6cf7 !important;
      background-color: rgba(74, 108, 247, 0.1) !important;
    }
    .graphiti-selected {
      outline: 2px solid #28a745 !important;
      background-color: rgba(40, 167, 69, 0.1) !important;
    }
  `;
  document.head.appendChild(style);
  
  // Track currently hovered element
  let currentHoveredElement = null;
  
  // Mouse move handler for hover effect
  function handleMouseMove(e) {
    // Remove highlight from previous element
    if (currentHoveredElement) {
      currentHoveredElement.classList.remove('graphiti-hover-highlight');
    }
    
    // Don't highlight the overlay or its children
    if (e.target === overlay || overlay.contains(e.target)) {
      return;
    }
    
    // Add highlight to current element
    currentHoveredElement = e.target;
    currentHoveredElement.classList.add('graphiti-hover-highlight');
  }
  
  // Click handler for selecting elements
  function handleClick(e) {
    // Don't select the overlay or its children
    if (e.target === overlay || overlay.contains(e.target)) {
      return;
    }
    
    // Toggle selection
    if (e.target.classList.contains('graphiti-selected')) {
      e.target.classList.remove('graphiti-selected');
      selectedElements = selectedElements.filter(el => el !== e.target);
    } else {
      e.target.classList.add('graphiti-selected');
      selectedElements.push(e.target);
    }
    
    // Update selected count
    document.getElementById('selected-count').textContent = `Selected: ${selectedElements.length} elements`;
    
    // Prevent default behavior
    e.preventDefault();
    e.stopPropagation();
  }
  
  // Finish selection
  function finishSelection() {
    // Clean up
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('click', handleClick);
    document.removeEventListener('keydown', handleKeyDown);
    
    // Remove hover highlight
    if (currentHoveredElement) {
      currentHoveredElement.classList.remove('graphiti-hover-highlight');
    }
    
    // Remove overlay
    document.body.removeChild(overlay);
    
    // Remove style
    document.head.removeChild(style);
    
    // Reset selection mode
    isSelectionMode = false;
    
    // Send selected elements to background script
    chrome.runtime.sendMessage({
      action: 'elementsSelected',
      count: selectedElements.length
    });
  }
  
  // Key handler for ESC key
  function handleKeyDown(e) {
    if (e.key === 'Escape') {
      finishSelection();
    }
  }
  
  // Add event listeners
  document.addEventListener('mousemove', handleMouseMove);
  document.addEventListener('click', handleClick);
  document.addEventListener('keydown', handleKeyDown);
  
  // Add finish button handler
  document.getElementById('finish-selection').addEventListener('click', finishSelection);
}
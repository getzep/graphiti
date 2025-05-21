document.addEventListener('DOMContentLoaded', function() {
  // DOM elements
  const connectionStatus = document.getElementById('connection-status');
  const extractDataBtn = document.getElementById('extract-data');
  const selectElementsBtn = document.getElementById('select-elements');
  const extractionResults = document.getElementById('extraction-results');
  const dataPreview = document.getElementById('data-preview');
  const aiCategories = document.getElementById('ai-categories');
  const customCategoryInput = document.getElementById('custom-category');
  const addCategoryBtn = document.getElementById('add-category');
  const customCategories = document.getElementById('custom-categories');
  const saveToGraphitiBtn = document.getElementById('save-to-graphiti');
  const apiEndpointInput = document.getElementById('api-endpoint');
  const apiKeyInput = document.getElementById('api-key');
  const saveSettingsBtn = document.getElementById('save-settings');

  // Load saved settings
  chrome.storage.local.get(['apiEndpoint', 'apiKey'], function(result) {
    if (result.apiEndpoint) {
      apiEndpointInput.value = result.apiEndpoint;
      checkConnection(result.apiEndpoint, result.apiKey);
    }
    if (result.apiKey) {
      apiKeyInput.value = result.apiKey;
    }
  });

  // Save settings
  saveSettingsBtn.addEventListener('click', function() {
    const apiEndpoint = apiEndpointInput.value.trim();
    const apiKey = apiKeyInput.value.trim();
    
    chrome.storage.local.set({
      apiEndpoint: apiEndpoint,
      apiKey: apiKey
    }, function() {
      checkConnection(apiEndpoint, apiKey);
      alert('Settings saved!');
    });
  });

  // Check connection to Graphiti API
  function checkConnection(apiEndpoint, apiKey) {
    if (!apiEndpoint) {
      updateConnectionStatus(false);
      return;
    }

    fetch(`${apiEndpoint}/healthcheck`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey && { 'Authorization': `Bearer ${apiKey}` })
      }
    })
    .then(response => {
      if (response.ok) {
        updateConnectionStatus(true);
      } else {
        updateConnectionStatus(false);
      }
    })
    .catch(error => {
      console.error('Connection error:', error);
      updateConnectionStatus(false);
    });
  }

  function updateConnectionStatus(isConnected) {
    if (isConnected) {
      connectionStatus.classList.remove('disconnected');
      connectionStatus.classList.add('connected');
      connectionStatus.querySelector('.status-text').textContent = 'Connected';
    } else {
      connectionStatus.classList.remove('connected');
      connectionStatus.classList.add('disconnected');
      connectionStatus.querySelector('.status-text').textContent = 'Disconnected';
    }
  }

  // Extract data from current page
  extractDataBtn.addEventListener('click', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      chrome.tabs.sendMessage(tabs[0].id, {action: "extractPageData"}, function(response) {
        if (response && response.data) {
          displayExtractedData(response.data);
          
          // Get AI categorization
          getAICategorization(response.data);
          
          extractionResults.classList.remove('hidden');
        } else {
          alert('Failed to extract data from the page.');
        }
      });
    });
  });

  // Select specific elements on the page
  selectElementsBtn.addEventListener('click', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      chrome.tabs.sendMessage(tabs[0].id, {action: "enableElementSelection"});
      window.close(); // Close popup to allow element selection
    });
  });

  // Display extracted data
  function displayExtractedData(data) {
    // Clear previous data
    dataPreview.innerHTML = '';
    
    // Create a formatted preview
    const preElement = document.createElement('pre');
    preElement.textContent = JSON.stringify(data, null, 2);
    dataPreview.appendChild(preElement);
  }

  // Get AI categorization for the extracted data
  function getAICategorization(data) {
    const apiEndpoint = apiEndpointInput.value.trim();
    const apiKey = apiKeyInput.value.trim();
    
    if (!apiEndpoint) {
      alert('Please set the Graphiti API endpoint in settings.');
      return;
    }
    
    // Clear previous categories
    aiCategories.innerHTML = '';
    
    // Show loading indicator
    const loadingEl = document.createElement('div');
    loadingEl.textContent = 'Analyzing data...';
    aiCategories.appendChild(loadingEl);
    
    fetch(`${apiEndpoint}/browser-agent/categorize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey && { 'Authorization': `Bearer ${apiKey}` })
      },
      body: JSON.stringify({ content: data })
    })
    .then(response => response.json())
    .then(data => {
      // Clear loading indicator
      aiCategories.innerHTML = '';
      
      // Display AI suggested categories
      const categories = data.categories || [];
      categories.forEach(category => {
        const categoryTag = createCategoryTag(category, false);
        aiCategories.appendChild(categoryTag);
      });
    })
    .catch(error => {
      console.error('Error getting AI categorization:', error);
      aiCategories.innerHTML = '<div class="error">Failed to get AI categorization</div>';
    });
  }

  // Add custom category
  addCategoryBtn.addEventListener('click', function() {
    const category = customCategoryInput.value.trim();
    if (category) {
      const categoryTag = createCategoryTag(category, true);
      customCategories.appendChild(categoryTag);
      customCategoryInput.value = '';
    }
  });

  // Create a category tag element
  function createCategoryTag(category, removable) {
    const categoryTag = document.createElement('div');
    categoryTag.className = 'category-tag';
    categoryTag.textContent = category;
    
    if (removable) {
      const removeBtn = document.createElement('span');
      removeBtn.className = 'remove-tag';
      removeBtn.innerHTML = '&times;';
      removeBtn.addEventListener('click', function() {
        categoryTag.remove();
      });
      categoryTag.appendChild(removeBtn);
    }
    
    return categoryTag;
  }

  // Save data to Graphiti knowledge graph
  saveToGraphitiBtn.addEventListener('click', function() {
    const apiEndpoint = apiEndpointInput.value.trim();
    const apiKey = apiKeyInput.value.trim();
    
    if (!apiEndpoint) {
      alert('Please set the Graphiti API endpoint in settings.');
      return;
    }
    
    // Get the extracted data
    const extractedData = JSON.parse(dataPreview.querySelector('pre').textContent);
    
    // Get all categories (AI + custom)
    const allCategories = [];
    
    // Get AI categories
    Array.from(aiCategories.querySelectorAll('.category-tag')).forEach(tag => {
      allCategories.push(tag.textContent);
    });
    
    // Get custom categories
    Array.from(customCategories.querySelectorAll('.category-tag')).forEach(tag => {
      // Remove the "×" character from the text content
      const categoryText = tag.textContent.replace('×', '').trim();
      allCategories.push(categoryText);
    });
    
    // Prepare data for saving
    const dataToSave = {
      content: extractedData,
      categories: allCategories,
      source: {
        url: extractedData.url || window.location.href,
        title: extractedData.title || document.title,
        timestamp: new Date().toISOString()
      }
    };
    
    // Send data to Graphiti API
    fetch(`${apiEndpoint}/browser-agent/save`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey && { 'Authorization': `Bearer ${apiKey}` })
      },
      body: JSON.stringify(dataToSave)
    })
    .then(response => {
      if (response.ok) {
        alert('Data successfully saved to Graphiti knowledge graph!');
        extractionResults.classList.add('hidden');
      } else {
        alert('Failed to save data to Graphiti.');
      }
    })
    .catch(error => {
      console.error('Error saving to Graphiti:', error);
      alert('Error saving to Graphiti: ' + error.message);
    });
  });
});
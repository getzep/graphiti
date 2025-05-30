// Listen for messages from content script
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === 'elementsSelected') {
    // Create a notification when elements are selected
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: 'Graphiti Web Scraper',
      message: `${request.count} elements selected. Open the extension to extract data.`
    });
  }
});

// Open the popup when the extension icon is clicked
chrome.action.onClicked.addListener(function(tab) {
  chrome.action.openPopup();
});
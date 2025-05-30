/**
 * Automated test script for the Graphiti Browser Agent extension.
 * This script tests the functionality of the browser extension components.
 * 
 * To use this script:
 * 1. Load the extension in Chrome developer mode
 * 2. Open the browser console
 * 3. Copy and paste this script into the console
 * 4. The tests will run automatically and report results
 */

class BrowserAgentExtensionTester {
  constructor() {
    this.testResults = [];
    this.testCount = 0;
    this.passCount = 0;
  }

  /**
   * Log a test result
   * @param {string} testName - Name of the test
   * @param {boolean} passed - Whether the test passed
   * @param {string} message - Additional message
   * @param {object} details - Test details
   */
  logResult(testName, passed, message = "", details = null) {
    this.testCount++;
    if (passed) this.passCount++;
    
    const result = {
      testName,
      passed,
      message,
      details,
      timestamp: new Date().toISOString()
    };
    
    this.testResults.push(result);
    
    // Log to console with color
    const status = passed ? "%cPASS" : "%cFAIL";
    const style = passed ? "color: green; font-weight: bold;" : "color: red; font-weight: bold;";
    
    console.log(`[${status}] ${testName}`, style);
    if (message) console.log(`  ${message}`);
    if (details) console.log("  Details:", details);
  }

  /**
   * Test manifest.json structure
   */
  async testManifest() {
    const testName = "Manifest Structure";
    
    try {
      const manifest = chrome.runtime.getManifest();
      
      // Check required fields
      const requiredFields = ["name", "version", "manifest_version", "action", "permissions"];
      const missingFields = requiredFields.filter(field => !(field in manifest));
      
      if (missingFields.length > 0) {
        this.logResult(testName, false, `Missing required fields: ${missingFields.join(", ")}`, manifest);
        return;
      }
      
      // Check permissions
      const requiredPermissions = ["activeTab", "storage"];
      const missingPermissions = requiredPermissions.filter(perm => !manifest.permissions.includes(perm));
      
      if (missingPermissions.length > 0) {
        this.logResult(testName, false, `Missing required permissions: ${missingPermissions.join(", ")}`, manifest.permissions);
        return;
      }
      
      this.logResult(testName, true, "Manifest structure is valid", {
        name: manifest.name,
        version: manifest.version,
        permissions: manifest.permissions
      });
    } catch (error) {
      this.logResult(testName, false, `Error testing manifest: ${error.message}`);
    }
  }

  /**
   * Test storage functionality
   */
  async testStorage() {
    const testName = "Storage Functionality";
    
    try {
      // Test data
      const testData = {
        apiEndpoint: "https://test-api.example.com",
        apiKey: "test-api-key-12345"
      };
      
      // Save to storage
      await new Promise((resolve, reject) => {
        chrome.storage.local.set(testData, () => {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else {
            resolve();
          }
        });
      });
      
      // Retrieve from storage
      const retrievedData = await new Promise((resolve, reject) => {
        chrome.storage.local.get(["apiEndpoint", "apiKey"], (result) => {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else {
            resolve(result);
          }
        });
      });
      
      // Verify data
      if (retrievedData.apiEndpoint !== testData.apiEndpoint || 
          retrievedData.apiKey !== testData.apiKey) {
        this.logResult(testName, false, "Retrieved data doesn't match saved data", {
          saved: testData,
          retrieved: retrievedData
        });
        return;
      }
      
      this.logResult(testName, true, "Storage functionality works correctly");
      
      // Clean up
      await new Promise(resolve => {
        chrome.storage.local.remove(["apiEndpoint", "apiKey"], resolve);
      });
    } catch (error) {
      this.logResult(testName, false, `Error testing storage: ${error.message}`);
    }
  }

  /**
   * Test message passing between popup and content script
   */
  async testMessagePassing() {
    const testName = "Message Passing";
    
    try {
      // This test requires an active tab
      const tabs = await new Promise((resolve) => {
        chrome.tabs.query({active: true, currentWindow: true}, resolve);
      });
      
      if (tabs.length === 0) {
        this.logResult(testName, false, "No active tab found for testing");
        return;
      }
      
      // Send a test message to the content script
      const response = await new Promise((resolve) => {
        chrome.tabs.sendMessage(tabs[0].id, {action: "testConnection"}, resolve);
      });
      
      if (!response || !response.success) {
        this.logResult(testName, false, "Content script did not respond correctly", response);
        return;
      }
      
      this.logResult(testName, true, "Message passing works correctly", response);
    } catch (error) {
      this.logResult(testName, false, `Error testing message passing: ${error.message}`);
    }
  }

  /**
   * Test DOM manipulation functions
   */
  async testDOMFunctions() {
    const testName = "DOM Functions";
    
    try {
      // Create a test container
      const container = document.createElement('div');
      container.id = 'test-container';
      document.body.appendChild(container);
      
      // Create a category tag
      const categoryTag = document.createElement('div');
      categoryTag.className = 'category-tag';
      categoryTag.textContent = 'Test Category';
      
      const removeBtn = document.createElement('span');
      removeBtn.className = 'remove-tag';
      removeBtn.innerHTML = '&times;';
      categoryTag.appendChild(removeBtn);
      
      container.appendChild(categoryTag);
      
      // Test removal functionality
      removeBtn.click();
      
      // Check if the tag was removed
      const tagsAfterRemoval = container.querySelectorAll('.category-tag');
      
      if (tagsAfterRemoval.length > 0) {
        this.logResult(testName, false, "Category tag was not removed after clicking remove button");
      } else {
        this.logResult(testName, true, "DOM manipulation functions work correctly");
      }
      
      // Clean up
      document.body.removeChild(container);
    } catch (error) {
      this.logResult(testName, false, `Error testing DOM functions: ${error.message}`);
    }
  }

  /**
   * Print a summary of all test results
   */
  printSummary() {
    console.log("\n" + "=".repeat(50));
    console.log(`%cTEST SUMMARY: ${this.passCount}/${this.testCount} tests passed`, 
                "font-weight: bold; font-size: 14px;");
    console.log("=".repeat(50) + "\n");
    
    if (this.passCount < this.testCount) {
      console.log("%cFailed Tests:", "color: red; font-weight: bold;");
      this.testResults.forEach(result => {
        if (!result.passed) {
          console.log(`- ${result.testName}: ${result.message}`);
        }
      });
    }
    
    return this.passCount === this.testCount;
  }

  /**
   * Run all tests
   */
  async runAllTests() {
    console.log("%cRunning Graphiti Browser Agent Extension Tests...", 
                "color: blue; font-weight: bold; font-size: 16px;");
    
    await this.testManifest();
    await this.testStorage();
    await this.testMessagePassing();
    await this.testDOMFunctions();
    
    return this.printSummary();
  }
}

// Run the tests
const tester = new BrowserAgentExtensionTester();
tester.runAllTests().then(allPassed => {
  console.log(allPassed ? 
    "%cAll tests passed! The extension is working correctly." : 
    "%cSome tests failed. Please check the issues above.", 
    `color: ${allPassed ? "green" : "red"}; font-weight: bold;`);
});
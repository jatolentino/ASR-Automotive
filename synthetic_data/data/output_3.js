function extractDTText() {
  const dtElements = document.querySelectorAll('dt');
  const texts = [];

  dtElements.forEach(dt => {
    // Get the text content inside the <dt> tag
    texts.push(dt.textContent.trim());
	console.log(dt.textContent.trim())
  });

  return texts;
}

// Example usage
const extractedTexts = extractDTText();
console.log(extractedTexts);  // Output: ["A/C compressor", "A-pillar", "ACV"]
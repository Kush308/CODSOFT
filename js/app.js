const captionTemplates = {
  nature: [
    "A serene forest landscape with tall evergreen trees reaching toward the sky",
    "Majestic mountains covered in lush green vegetation under a clear blue sky",
    "A peaceful lake surrounded by dense woodland and rocky outcrops",
    "Sunlight filtering through the canopy of an ancient forest"
  ],
  city: [
    "A bustling urban cityscape with towering skyscrapers and busy streets",
    "Modern architecture dominates the skyline in this metropolitan area",
    "City lights illuminate the evening sky in this urban landscape",
    "A vibrant downtown area with glass buildings reflecting the sunset"
  ],
  animal: [
    "A curious orange tabby cat sitting peacefully by a sunny window",
    "An adorable feline companion resting comfortably in natural light",
    "A domestic cat with bright eyes gazing thoughtfully outside",
    "A well-groomed pet cat enjoying a quiet moment indoors"
  ],
  food: [
    "A freshly baked pizza topped with melted cheese and colorful vegetables",
    "Delicious Italian cuisine featuring crispy crust and premium ingredients",
    "A mouth-watering meal with perfectly balanced flavors and textures",
    "Artisanal pizza with fresh herbs and high-quality toppings"
  ],
  general: [
    "An interesting scene captured with excellent composition and lighting",
    "A well-framed photograph showcasing beautiful visual elements",
    "A compelling image with rich details and vibrant colors",
    "A thoughtfully composed photograph with engaging subject matter"
  ]
};

let currentImage = null;
let processingStartTime = null;

// File input handling
document.getElementById('fileInput').addEventListener('change', handleFileSelect);

// Drag and drop functionality
const uploadSection = document.getElementById('uploadSection');
uploadSection.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadSection.classList.add('dragover');
});
uploadSection.addEventListener('dragleave', () => {
  uploadSection.classList.remove('dragover');
});
uploadSection.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadSection.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
});

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
    handleFile(file);
  }
}
function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select a valid image file.');
    return;
  }
  const reader = new FileReader();
  reader.onload = function(e) {
    const imagePreview = document.getElementById('imagePreview');
    imagePreview.src = e.target.result;
    imagePreview.style.display = 'block';
    currentImage = e.target.result;
    // Update image resolution
    const img = new Image();
    img.onload = function() {
      document.getElementById('imageResolution').textContent = `${this.width} × ${this.height}px`;
    };
    img.src = e.target.result;
    // Start processing
    setTimeout(() => processImage(), 1000);
  };
  reader.readAsDataURL(file);
}

function processImage() {
  processingStartTime = Date.now();
  document.getElementById('processingAnimation').classList.add('active');
  document.getElementById('captionResult').classList.remove('show');
  const steps = ['step1', 'step2', 'step3', 'step4', 'step5'];
  let currentStep = 0;
  const stepInterval = setInterval(() => {
    if (currentStep < steps.length) {
      document.getElementById(steps[currentStep]).classList.add('active');
      currentStep++;
    } else {
      clearInterval(stepInterval);
      setTimeout(() => showResults(), 500);
    }
  }, 800);
}

function showResults() {
  document.getElementById('processingAnimation').classList.remove('active');
  const steps = document.querySelectorAll('.step');
  steps.forEach(step => step.classList.remove('active'));
  const caption = generateCaption();
  const confidence = Math.floor(Math.random() * 15) + 85;
  document.getElementById('captionText').textContent = caption;
  document.getElementById('confidenceScore').textContent = `Confidence: ${confidence}%`;
  const processingTime = ((Date.now() - processingStartTime) / 1000).toFixed(1);
  document.getElementById('processingTime').textContent = `${processingTime}s`;
  document.getElementById('captionResult').classList.add('show');
}

function generateCaption(imageType = 'general') {
  const templates = captionTemplates[imageType] || captionTemplates.general;
  return templates[Math.floor(Math.random() * templates.length)];
}

function loadExample(type) {
  const examples = {
    nature: '🌲 Forest Example',
    city: '🏙 City Example',
    animal: '🐱 Cat Example',
    food: '🍕 Pizza Example'
  };
  const imagePreview = document.getElementById('imagePreview');
  imagePreview.style.display = 'block';
  imagePreview.src = `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"><rect width="400" height="300" fill="#f0f0f0"/><text x="200" y="150" text-anchor="middle" font-size="24" fill="#666">${examples[type]}</text></svg>`;
  currentImage = type;
  document.getElementById('imageResolution').textContent = '400 × 300px';
  setTimeout(() => {
    processingStartTime = Date.now();
    processImageWithType(type);
  }, 500);
}

function processImageWithType(type) {
  document.getElementById('processingAnimation').classList.add('active');
  document.getElementById('captionResult').classList.remove('show');
  const steps = ['step1', 'step2', 'step3', 'step4', 'step5'];
  let currentStep = 0;
  const stepInterval = setInterval(() => {
    if (currentStep < steps.length) {
      document.getElementById(steps[currentStep]).classList.add('active');
      currentStep++;
    } else {
      clearInterval(stepInterval);
      setTimeout(() => showResultsWithType(type), 500);
    }
  }, 600);
}

function showResultsWithType(type) {
  document.getElementById('processingAnimation').classList.remove('active');
  const steps = document.querySelectorAll('.step');
  steps.forEach(step => step.classList.remove('active'));
  const caption = generateCaption(type);
  const confidence = Math.floor(Math.random() * 10) + 90;
  document.getElementById('captionText').textContent = caption;
  document.getElementById('confidenceScore').textContent = `Confidence: ${confidence}%`;
  const processingTime = ((Date.now() - processingStartTime) / 1000).toFixed(1);
  document.getElementById('processingTime').textContent = `${processingTime}s`;
  document.getElementById('captionResult').classList.add('show');
}

function generateAlternative() {
  const currentType = typeof currentImage === 'string' ? currentImage : 'general';
  const newCaption = generateCaption(currentType);
  const newConfidence = Math.floor(Math.random() * 15) + 85;
  document.getElementById('captionText').textContent = newCaption;
  document.getElementById('confidenceScore').textContent = `Confidence: ${newConfidence}%`;
  const captionResult = document.getElementById('captionResult');
  captionResult.style.animation = 'none';
  setTimeout(() => {
    captionResult.style.animation = 'slideIn 0.5s ease';
  }, 10);
}

function copyCaption() {
  const captionText = document.getElementById('captionText').textContent;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(captionText).then(() => {
      showCopyFeedback();
    }).catch(() => {
      fallbackCopy(captionText);
    });
  } else {
    fallbackCopy(captionText);
  }
}

function fallbackCopy(text) {
  const textArea = document.createElement('textarea');
  textArea.value = text;
  textArea.style.position = 'fixed';
  textArea.style.opacity = '0';
  document.body.appendChild(textArea);
  textArea.select();
  try {
    document.execCommand('copy');
    showCopyFeedback();
  } catch (err) {
    alert('Caption: ' + text);
  }
  document.body.removeChild(textArea);
}

function showCopyFeedback() {
  const btn = event.target;
  const originalText = btn.textContent;
  btn.textContent = '✅ Copied!';
  btn.style.background = '#4caf50';
  setTimeout(() => {
    btn.textContent = originalText;
    btn.style.background = '';
  }, 2000);
}

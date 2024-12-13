const fs = require('fs');
const path = require('path');

// Directory where images are stored
const imageDirectory = path.join(__dirname, 'archive/images/images'); // Adjust if necessary

// Recursively scan directories for image files
function getImages(dir) {
    let images = [];
    const items = fs.readdirSync(dir);
    items.forEach(item => {
        const itemPath = path.join(dir, item);
        const stat = fs.statSync(itemPath);
        if (stat.isDirectory()) {
            // If it's a directory, scan recursively
            images = images.concat(getImages(itemPath));
        } else if (stat.isFile() && /\.(jpg|jpeg|png|gif|bmp)$/i.test(item)) {
            // If it's an image file (you can add more extensions if needed)
            images.push(itemPath.replace(__dirname, '')); // Store relative path
        }
    });
    return images;
}

// Get all image paths
const images = getImages(imageDirectory);

// Create the JSON file
const jsonFilePath = path.join(__dirname, 'images.json');
fs.writeFileSync(jsonFilePath, JSON.stringify(images, null, 2), 'utf8');

console.log('images.json file created!');

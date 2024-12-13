let imageIndex = parseInt(localStorage.getItem("imageIndex")) || 0; // Load imageIndex from localStorage (or start from 0)
let images = [];
let votes = JSON.parse(localStorage.getItem("votes")) || []; // Load existing votes

const currentImage = document.getElementById("currentImage");
const voteButtons = document.querySelectorAll("button"); // Get vote buttons

// Load images from JSON
fetch('images.json')
  .then(response => response.json())
  .then(data => {
    images = data;
    filterImages();
    if (images.length > 0) {
      displayImage();
    } else {
      alert("No more images to vote on!");
    }
  })
  .catch(error => console.error('Error loading images:', error));

// Filter out already voted images
function filterImages() {
  const votedPaths = votes.map(v => v.split(" ")[1]);
  images = images.filter(image => !votedPaths.includes(image));
}

// Display the current image
function displayImage() {
  if (imageIndex < images.length) {
    currentImage.src = images[imageIndex];
    currentImage.hidden = false;
    voteButtons.forEach(button => button.hidden = false);  // Show vote buttons
  } else {
    alert("You've voted on all images!");
    downloadCSV();
  }
}

// Handle voting and persist to localStorage
function vote(option) {
  const fileName = images[imageIndex];
  const voteValue = option === 'yes' ? `1 ${fileName}` : `0 ${fileName}`;
  votes.push(voteValue);
  localStorage.setItem("votes", JSON.stringify(votes)); // Save votes to localStorage
  imageIndex++;  // Move to next image
  localStorage.setItem("imageIndex", imageIndex); // Save current imageIndex
  displayImage();
}

// Download CSV when voting is complete
function downloadCSV() {
  let csvContent = "Vote,Image\n";
  votes.forEach(vote => {
    const [voteValue, path] = vote.split(" ");
    csvContent += `${voteValue},${path}\n`;
  });

  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'votes.csv';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

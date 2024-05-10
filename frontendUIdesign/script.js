document.querySelectorAll('.market-button').forEach(button => {
    button.addEventListener('click', () => {
      const market = button.innerText;
      // Send market data to your scraper or perform any other action
      document.getElementById('output').innerText = `Selected Market: ${market}`;
    });
  });
  
  document.querySelectorAll('.trade-button').forEach(button => {
    button.addEventListener('click', () => {
      const tradeType = button.innerText;
      // Send trade type data to your scraper or perform any other action
      document.getElementById('output').innerText = `Selected Trade Type: ${tradeType}`;
    });
  });
   // JavaScript code for toggling the menu
   function toggleMenu() {
    var menu = document.getElementById("menu");
    if (menu.style.display === "block") {
        menu.style.display = "none";
    } else {
        menu.style.display = "block";
    }
};


// Scroll to section function
function scrollToSection(id, event) {
    event.preventDefault();
    const target = document.getElementById(id);
    target.scrollIntoView({ behavior: 'smooth' });
};


// Add click event listener to the button
document.getElementById('signup-btn').addEventListener('click', function() {
    // Redirect to the login page
    window.location.href = 'log.html';
  });
  
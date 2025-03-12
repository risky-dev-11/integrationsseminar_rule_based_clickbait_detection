// Function to handle sending messages
function sendMessage() {
    const chatInput = document.getElementById('chat_input');
    const message = chatInput.value.trim();
    
    if (message) {
        // Append user message
        appendMessage(message, 'user');
        
        // Clear input field
        chatInput.value = '';
        
        // Process message and get bot response
        processMessage(message);
    }
}

// Function to append a message to the chat
function appendMessage(text, sender) {
    const chat = document.getElementById('chat');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat_message ${sender}`;
    
    const messageTextDiv = document.createElement('div');
    messageTextDiv.className = 'chat_message_text';
    messageTextDiv.textContent = text;
    
    messageDiv.appendChild(messageTextDiv);
    chat.appendChild(messageDiv);
    
    // Scroll to bottom of chat
    chat.scrollTop = chat.scrollHeight;
}

// Function to process user message and respond
function processMessage(message) {
    // Simulate a bot response (replace with actual processing logic)
    setTimeout(() => {
        // Example simple response - in a real app, this would be replaced with actual logic
        let botResponse = "Ich analysiere deinen Text...";
        
        appendMessage(botResponse, 'bot');
    }, 600); // Small delay to simulate processing
}

// Event listener for Enter key in the input field
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat_input');
    chatInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});

// Function to update SVG color based on input content
function updateSendButtonColor() {
    const chatInput = document.getElementById('chat_input');
    const sendButton = document.getElementById('chat_send');
    const svg = sendButton.querySelector('svg path');
    
    if (chatInput.value.trim() === '') {
        svg.setAttribute('fill', getComputedStyle(document.documentElement).getPropertyValue('--border-color'));
    } else {
        svg.setAttribute('fill', getComputedStyle(document.documentElement).getPropertyValue('--accent-color'));
    }
}

// Add event listeners to update button color
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat_input');
    chatInput.addEventListener('input', updateSendButtonColor);
    
    // Set initial color
    updateSendButtonColor();
});
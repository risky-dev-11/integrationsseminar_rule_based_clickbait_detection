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

const percentFormatter = new Intl.NumberFormat('de-DE', {
    style: 'percent',
    minimumFractionDigits: 2,
});

const modelNameMap = {
    neural: 'Neuronales Netz',
    rule_based: 'Regelbasierter Klassifikator',
};

const formatResponse = data => {
    const isClickbait = data.is_clickbait ? 'Clickbait' : 'kein Clickbait';
    const title = `[${modelNameMap[data.model]}] ${isClickbait}:`;

    let details = `Dein Text "${data.headline}" ist`;
    if (data.model === 'neural') {
        const formattedProbability = percentFormatter.format(data.probability);

        details += ` zu ${formattedProbability} Clickbait!`;
    } else {
        details += ` ${isClickbait}!`;
    }

    return [title, details].join('\r\n');
};

// Function to process user message and respond
function processMessage(message) {
    // Get the selected model
    const modelSelect = document.getElementById('model_select');
    const selectedModel = modelSelect.value;

    // Show loading message
    appendMessage('Ich analysiere deinen Text...', 'bot');

    // Send API request to the backend with the selected model
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `message=${encodeURIComponent(
            message
        )}&model=${encodeURIComponent(selectedModel)}`,
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Netzwerk-Antwort war nicht ok');
            }
            return response.json();
        })
        .then(data => {
            // Remove loading message
            const chatElement = document.getElementById('chat');
            const loadingMessage = chatElement.lastChild;
            chatElement.removeChild(loadingMessage);

            // Create response based on prediction
            if (data.error) {
                appendMessage(`Fehler bei der Analyse: ${data.error}`, 'bot');
                return;
            }

            if (data.model === 'llm') {
                appendMessage(`${data.answer}`, 'bot');
                return;
            }

            appendMessage(formatResponse(data), 'bot');
        })
        .catch(error => {
            console.error('Fehler bei der Anfrage:', error);
            appendMessage(
                'Es ist ein Fehler aufgetreten. Bitte versuche es später noch einmal.',
                'bot'
            );
        });
}

// Event listener for Enter key in the input field
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat_input');
    chatInput.addEventListener('keypress', function (event) {
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
        svg.setAttribute(
            'fill',
            getComputedStyle(document.documentElement).getPropertyValue(
                '--border-color'
            )
        );
    } else {
        svg.setAttribute(
            'fill',
            getComputedStyle(document.documentElement).getPropertyValue(
                '--accent-color'
            )
        );
    }
}

// Add event listeners to update button color
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat_input');
    chatInput.addEventListener('input', updateSendButtonColor);

    // Set initial color
    updateSendButtonColor();
});

// Chatbot Page Functionality

document.addEventListener('DOMContentLoaded', function() {
    initializeChatInterface();
    initializeChatSettings();
    initializeChatAnalytics();
    initializeVoiceInput();
    initializeQuickActions();
    initializeExampleConversations();
    initializeAOS();
    updateCurrentTime();
});

// Global chat state
const chatState = {
    messages: [],
    sessionStartTime: Date.now(),
    topicsDiscussed: new Set(),
    messageCount: 0,
    isTyping: false,
    personality: 'professional',
    responseLength: 'detailed',
    expertiseFocus: 'all',
    includeCode: true,
    includeCitations: false
};

// Main chat interface
function initializeChatInterface() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-message');
    const chatMessages = document.getElementById('chat-messages');

    if (!chatInput || !sendButton) return;

    // Send message functionality
    const sendMessage = () => {
        const message = chatInput.value.trim();
        if (!message || chatState.isTyping) return;

        addUserMessage(message);
        chatInput.value = '';
        sendButton.disabled = true;

        // Simulate AI response
        setTimeout(() => {
            const response = generateAIResponse(message);
            addAIMessage(response);
            sendButton.disabled = false;
        }, 1000 + Math.random() * 2000);
    };

    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Suggestion chips functionality
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('suggestion-chip')) {
            const suggestion = e.target.getAttribute('data-suggestion');
            if (suggestion) {
                chatInput.value = suggestion;
                sendMessage();
            }
        }
    });

    // Auto-resize input
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
}

// Chat settings management
function initializeChatSettings() {
    const personalitySelect = document.getElementById('ai-personality');
    const responseLengthSelect = document.getElementById('response-length');
    const expertiseFocusSelect = document.getElementById('expertise-focus');
    const codeExamplesToggle = document.getElementById('code-examples');
    const citationsToggle = document.getElementById('research-citations');
    const exportButton = document.getElementById('export-chat');
    const clearButton = document.getElementById('clear-chat');

    // Settings change handlers
    if (personalitySelect) {
        personalitySelect.addEventListener('change', function() {
            chatState.personality = this.value;
            addSystemMessage(`AI personality changed to: ${this.value}`);
        });
    }

    if (responseLengthSelect) {
        responseLengthSelect.addEventListener('change', function() {
            chatState.responseLength = this.value;
            addSystemMessage(`Response length set to: ${this.value}`);
        });
    }

    if (expertiseFocusSelect) {
        expertiseFocusSelect.addEventListener('change', function() {
            chatState.expertiseFocus = this.value;
            addSystemMessage(`Expertise focus: ${this.value === 'all' ? 'All areas' : this.value.toUpperCase()}`);
        });
    }

    if (codeExamplesToggle) {
        codeExamplesToggle.addEventListener('change', function() {
            chatState.includeCode = this.checked;
        });
    }

    if (citationsToggle) {
        citationsToggle.addEventListener('change', function() {
            chatState.includeCitations = this.checked;
        });
    }

    // Export functionality
    if (exportButton) {
        exportButton.addEventListener('click', exportConversation);
    }

    // Clear chat functionality
    if (clearButton) {
        clearButton.addEventListener('click', function() {
            if (confirm('Are you sure you want to clear the chat history?')) {
                clearChatHistory();
            }
        });
    }
}

// Chat analytics
function initializeChatAnalytics() {
    updateAnalytics();
    
    // Update analytics every 30 seconds
    setInterval(updateAnalytics, 30000);
}

function updateAnalytics() {
    const messageCountEl = document.getElementById('message-count');
    const sessionTimeEl = document.getElementById('session-time');
    const topicsCoveredEl = document.getElementById('topics-covered');
    const topicTagsEl = document.getElementById('topic-tags');
    const insightsListEl = document.getElementById('insights-list');

    if (messageCountEl) {
        messageCountEl.textContent = chatState.messageCount;
    }

    if (sessionTimeEl) {
        const sessionMinutes = Math.floor((Date.now() - chatState.sessionStartTime) / 60000);
        sessionTimeEl.textContent = `${sessionMinutes}m`;
    }

    if (topicsCoveredEl) {
        topicsCoveredEl.textContent = chatState.topicsDiscussed.size;
    }

    if (topicTagsEl) {
        topicTagsEl.innerHTML = '';
        Array.from(chatState.topicsDiscussed).forEach(topic => {
            const tag = document.createElement('span');
            tag.className = 'topic-tag';
            tag.textContent = topic;
            topicTagsEl.appendChild(tag);
        });
    }

    if (insightsListEl) {
        updateConversationInsights(insightsListEl);
    }
}

function updateConversationInsights(container) {
    const insights = [];

    if (chatState.messageCount > 5) {
        insights.push('Active conversation in progress');
    }

    if (chatState.topicsDiscussed.has('machine learning')) {
        insights.push('Strong focus on ML concepts');
    }

    if (chatState.topicsDiscussed.has('code')) {
        insights.push('Technical implementation discussed');
    }

    if (chatState.messageCount > 10) {
        insights.push('Deep dive conversation');
    }

    container.innerHTML = '';
    
    if (insights.length === 0) {
        insights.push('Start chatting to see insights');
    }

    insights.forEach(insight => {
        const item = document.createElement('div');
        item.className = 'insight-item';
        item.innerHTML = `
            <i class="fas fa-lightbulb"></i>
            <span>${insight}</span>
        `;
        container.appendChild(item);
    });
}

// Voice input functionality
function initializeVoiceInput() {
    const voiceToggle = document.getElementById('voice-toggle');
    const voiceModal = document.getElementById('voice-modal');
    const stopVoiceButton = document.getElementById('stop-voice');

    if (!voiceToggle) return;

    let recognition;
    let isListening = false;

    // Check for speech recognition support
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            document.getElementById('chat-input').value = transcript;
            closeVoiceModal();
        };

        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            closeVoiceModal();
        };

        recognition.onend = function() {
            isListening = false;
            closeVoiceModal();
        };
    }

    voiceToggle.addEventListener('click', function() {
        if (recognition && !isListening) {
            openVoiceModal();
            recognition.start();
            isListening = true;
        } else {
            showNotification('Speech recognition not supported in this browser', 'error');
        }
    });

    if (stopVoiceButton) {
        stopVoiceButton.addEventListener('click', function() {
            if (recognition && isListening) {
                recognition.stop();
            }
            closeVoiceModal();
        });
    }

    if (voiceModal) {
        voiceModal.addEventListener('click', function(e) {
            if (e.target === voiceModal) {
                if (recognition && isListening) {
                    recognition.stop();
                }
                closeVoiceModal();
            }
        });
    }
}

function openVoiceModal() {
    const modal = document.getElementById('voice-modal');
    if (modal) {
        modal.classList.add('active');
    }
}

function closeVoiceModal() {
    const modal = document.getElementById('voice-modal');
    if (modal) {
        modal.classList.remove('active');
    }
}

// Quick actions
function initializeQuickActions() {
    const quickActions = document.querySelectorAll('.quick-action');
    
    quickActions.forEach(action => {
        action.addEventListener('click', function() {
            const actionType = this.getAttribute('data-action');
            handleQuickAction(actionType);
        });
    });
}

function handleQuickAction(actionType) {
    const prompts = {
        'explain-concept': 'Can you explain a complex AI concept in simple terms?',
        'code-example': 'Show me a code example for implementing a neural network',
        'research-summary': 'What are the latest developments in transformer architectures?',
        'career-advice': 'What advice do you have for someone starting in machine learning?'
    };

    const prompt = prompts[actionType];
    if (prompt) {
        document.getElementById('chat-input').value = prompt;
        // Trigger send
        document.getElementById('send-message').click();
    }
}

// Example conversations
function initializeExampleConversations() {
    const exampleButtons = document.querySelectorAll('.try-example');
    
    exampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const exampleType = this.getAttribute('data-example');
            loadExampleConversation(exampleType);
        });
    });
}

function loadExampleConversation(exampleType) {
    const examples = {
        'attention-mechanisms': {
            user: "Can you explain how attention mechanisms work in transformers?",
            ai: "Attention mechanisms in transformers allow the model to focus on different parts of the input sequence when processing each element. Unlike traditional RNNs that process sequences sequentially, attention enables parallel processing and direct connections between any two positions in a sequence.\n\nThe key insight is that when processing a word, the model can 'attend' to all other words in the sequence, giving different weights to each based on their relevance. This is computed through three learned representations: Query (Q), Key (K), and Value (V).\n\nThe attention score is calculated as: Attention(Q,K,V) = softmax(QK^T/√d_k)V\n\nThis mechanism has revolutionized NLP by enabling models to capture long-range dependencies more effectively than previous architectures."
        },
        'cnn-implementation': {
            user: "Show me how to implement a CNN for image classification",
            ai: "Here's a PyTorch implementation of a CNN for image classification:\n\n```python\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass SimpleCNN(nn.Module):\n    def __init__(self, num_classes=10):\n        super(SimpleCNN, self).__init__()\n        # Convolutional layers\n        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)\n        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)\n        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)\n        \n        # Pooling layer\n        self.pool = nn.MaxPool2d(2, 2)\n        \n        # Fully connected layers\n        self.fc1 = nn.Linear(128 * 4 * 4, 512)\n        self.fc2 = nn.Linear(512, num_classes)\n        self.dropout = nn.Dropout(0.5)\n        \n    def forward(self, x):\n        # Feature extraction\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = self.pool(F.relu(self.conv3(x)))\n        \n        # Flatten for fully connected layers\n        x = x.view(-1, 128 * 4 * 4)\n        \n        # Classification\n        x = F.relu(self.fc1(x))\n        x = self.dropout(x)\n        x = self.fc2(x)\n        \n        return x\n```\n\nThis architecture follows the typical CNN pattern: convolution → activation → pooling, repeated multiple times, followed by fully connected layers for classification."
        },
        'ai-music': {
            user: "Tell me about your approach to AI-generated music",
            ai: "My approach to AI music generation combines traditional music theory with machine learning. I use several techniques:\n\n1. **Generative Adversarial Networks (GANs)**: For creating novel musical patterns and structures\n2. **Transformer models**: Adapted for sequential music generation, treating musical notes as tokens\n3. **Variational Autoencoders (VAEs)**: For exploring the latent space of musical compositions\n\nThe process involves:\n- Converting audio to MIDI representations\n- Training on diverse musical datasets\n- Incorporating music theory constraints\n- Post-processing for musical coherence\n\nI find that the most interesting results come from hybrid approaches that blend algorithmic composition with human creativity, rather than fully automated generation."
        }
    };

    const example = examples[exampleType];
    if (example) {
        // Clear current chat and add example
        addUserMessage(example.user);
        setTimeout(() => {
            addAIMessage(example.ai);
        }, 1000);
    }
}

// Message handling functions
function addUserMessage(message) {
    addMessage(message, 'user');
    chatState.messageCount++;
    extractTopics(message);
    updateAnalytics();
}

function addAIMessage(message) {
    addMessage(message, 'ai');
    chatState.messageCount++;
    extractTopics(message);
    updateAnalytics();
}

function addSystemMessage(message) {
    addMessage(message, 'system');
}

function addMessage(content, type) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const avatar = type === 'user' ? 'fa-user' : type === 'system' ? 'fa-cog' : 'fa-robot';
    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas ${avatar}"></i>
        </div>
        <div class="message-content">
            <div class="message-text">
                ${type === 'system' ? `<em>${content}</em>` : formatMessageContent(content)}
            </div>
            <div class="message-time">
                <span>${time}</span>
            </div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Store message
    chatState.messages.push({
        content,
        type,
        timestamp: Date.now()
    });

    // Show typing indicator for AI responses
    if (type === 'user') {
        showTypingIndicator();
    } else {
        hideTypingIndicator();
    }
}

function formatMessageContent(content) {
    // Format code blocks
    content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, 
        '<pre><code class="language-$1">$2</code></pre>');
    
    // Format inline code
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Format line breaks
    content = content.replace(/\n/g, '<br>');
    
    return content;
}

function showTypingIndicator() {
    const statusText = document.querySelector('.status-text');
    const typingIndicator = document.querySelector('.typing-indicator');
    
    if (statusText && typingIndicator) {
        statusText.style.display = 'none';
        typingIndicator.classList.add('active');
        chatState.isTyping = true;
    }
}

function hideTypingIndicator() {
    const statusText = document.querySelector('.status-text');
    const typingIndicator = document.querySelector('.typing-indicator');
    
    if (statusText && typingIndicator) {
        statusText.style.display = 'block';
        typingIndicator.classList.remove('active');
        chatState.isTyping = false;
    }
}

// AI response generation
function generateAIResponse(userMessage) {
    const responses = {
        greeting: [
            "Hello! I'm excited to discuss AI and machine learning with you. What would you like to explore?",
            "Hi there! I'm here to help with any questions about my work in AI, research, or creative projects.",
            "Welcome! Feel free to ask me about machine learning, my research, or any technical topics you're curious about."
        ],
        experience: [
            "I have over 5 years of experience in Applied AI and Machine Learning, specializing in computer vision, NLP, and deep learning architectures. I've worked on projects ranging from medical image analysis to real-time recommendation systems, and I'm particularly passionate about the intersection of AI and creativity.",
            "My journey in AI started with traditional machine learning and evolved into deep learning and neural networks. I've deployed models in production environments serving millions of users, and I love exploring how AI can enhance human creativity through music and video projects."
        ],
        skills: [
            "My technical expertise spans Python, TensorFlow, PyTorch, scikit-learn, and OpenCV. I'm proficient in MLOps, cloud platforms like AWS and GCP, and I have experience with distributed computing using Apache Spark. I also enjoy working with creative technologies for music generation and video processing.",
            "I specialize in computer vision, natural language processing, and generative models. My recent work focuses on transformer architectures, attention mechanisms, and multimodal AI systems. I'm also experienced in model optimization, deployment strategies, and building scalable ML pipelines."
        ],
        music: [
            "Music is my creative outlet! I compose electronic and ambient pieces using AI-assisted tools. My compositions blend algorithmic generation with human creativity, exploring the intersection of technology and art. I use GANs, transformers, and custom neural networks to generate musical patterns while maintaining artistic control.",
            "I'm fascinated by AI music generation because it represents the perfect fusion of technical expertise and creative expression. My approach involves training models on diverse musical datasets and then collaborating with the AI to create unique compositions that neither human nor machine could produce alone."
        ],
        code: [
            "I'd be happy to show you some code! Here's a simple example of a neural network for image classification:\n\n```python\nimport torch.nn as nn\n\nclass SimpleNet(nn.Module):\n    def __init__(self, num_classes=10):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(3, 64, 3, padding=1),\n            nn.ReLU(),\n            nn.MaxPool2d(2),\n            nn.Conv2d(64, 128, 3, padding=1),\n            nn.ReLU(),\n            nn.MaxPool2d(2)\n        )\n        self.classifier = nn.Linear(128 * 8 * 8, num_classes)\n    \n    def forward(self, x):\n        x = self.features(x)\n        x = x.view(x.size(0), -1)\n        return self.classifier(x)\n```\n\nWhat specific aspect would you like me to explain further?"
        ],
        research: [
            "My current research focuses on multimodal AI systems that can understand and generate content across different modalities - text, images, audio, and video. I'm particularly interested in how attention mechanisms can be adapted for cross-modal understanding and how we can build more interpretable AI systems.",
            "I'm working on several exciting projects including real-time neural style transfer for video, few-shot learning for music genre classification, and explainable AI techniques for medical image analysis. The goal is always to create AI that enhances human capabilities rather than replacing human judgment."
        ],
        default: [
            "That's an interesting question! Could you be more specific? I can discuss my technical experience, research projects, creative work, or provide insights into machine learning concepts. What aspect would you like to explore?",
            "I'd love to help you with that! You can ask me about machine learning algorithms, my research projects, coding examples, music composition, video production, or career advice in AI. What interests you most?",
            "Great question! I'm here to share insights about AI, machine learning, and creative technology. Whether you're interested in technical details, practical applications, or creative projects, I'm ready to dive deep into any topic."
        ]
    };

    // Simple keyword matching for demonstration
    const message = userMessage.toLowerCase();
    let responseCategory = 'default';

    if (message.includes('hello') || message.includes('hi') || message.includes('hey')) {
        responseCategory = 'greeting';
    } else if (message.includes('experience') || message.includes('background') || message.includes('career')) {
        responseCategory = 'experience';
    } else if (message.includes('skill') || message.includes('technology') || message.includes('tech stack')) {
        responseCategory = 'skills';
    } else if (message.includes('music') || message.includes('compose') || message.includes('sound')) {
        responseCategory = 'music';
    } else if (message.includes('code') || message.includes('programming') || message.includes('implementation')) {
        responseCategory = 'code';
    } else if (message.includes('research') || message.includes('project') || message.includes('study')) {
        responseCategory = 'research';
    }

    const categoryResponses = responses[responseCategory];
    const response = categoryResponses[Math.floor(Math.random() * categoryResponses.length)];

    return response;
}

function extractTopics(message) {
    const topicKeywords = {
        'machine learning': ['machine learning', 'ml', 'model', 'training', 'algorithm'],
        'deep learning': ['deep learning', 'neural network', 'cnn', 'rnn', 'lstm', 'transformer'],
        'computer vision': ['computer vision', 'cv', 'image', 'detection', 'classification', 'opencv'],
        'nlp': ['nlp', 'natural language', 'text', 'language model', 'sentiment'],
        'code': ['code', 'programming', 'python', 'implementation', 'function'],
        'research': ['research', 'paper', 'study', 'experiment', 'analysis'],
        'music': ['music', 'composition', 'audio', 'sound', 'melody'],
        'video': ['video', 'film', 'visual', 'editing', 'production']
    };

    const text = message.toLowerCase();
    Object.entries(topicKeywords).forEach(([topic, keywords]) => {
        if (keywords.some(keyword => text.includes(keyword))) {
            chatState.topicsDiscussed.add(topic);
        }
    });
}

// Utility functions
function exportConversation() {
    const conversation = {
        timestamp: new Date().toISOString(),
        sessionDuration: Date.now() - chatState.sessionStartTime,
        messageCount: chatState.messageCount,
        topicsDiscussed: Array.from(chatState.topicsDiscussed),
        messages: chatState.messages,
        settings: {
            personality: chatState.personality,
            responseLength: chatState.responseLength,
            expertiseFocus: chatState.expertiseFocus,
            includeCode: chatState.includeCode,
            includeCitations: chatState.includeCitations
        }
    };

    const blob = new Blob([JSON.stringify(conversation, null, 2)], {
        type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ai-conversation-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification('Conversation exported successfully!', 'success');
}

function clearChatHistory() {
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        // Remove all messages except the welcome message
        const welcomeMessage = chatMessages.querySelector('.message.ai-message');
        chatMessages.innerHTML = '';
        if (welcomeMessage) {
            chatMessages.appendChild(welcomeMessage);
        }
    }

    // Reset state
    chatState.messages = [];
    chatState.messageCount = 0;
    chatState.topicsDiscussed.clear();
    chatState.sessionStartTime = Date.now();

    updateAnalytics();
    showNotification('Chat history cleared', 'info');
}

function updateCurrentTime() {
    const timeElement = document.getElementById('current-time');
    if (timeElement) {
        timeElement.textContent = new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'exclamation' : 'info'}-circle"></i>
            <span>${message}</span>
        </div>
    `;
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#2563eb'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        z-index: 3000;
        animation: slideInRight 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

function initializeAOS() {
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 800,
            once: true,
            offset: 100
        });
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
`;
document.head.appendChild(style);

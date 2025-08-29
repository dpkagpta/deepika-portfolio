// Website Interactivity and Functionality

document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeChatbot();
    initializeMusicPlayer();
    initializeVideoPlayer();
    initializeBlogInteractions();
    initializeScrollEffects();
    initializeMobileMenu();
});

// Navigation functionality
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');

    // Smooth scrolling for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                const headerHeight = document.querySelector('.header').offsetHeight;
                const targetPosition = targetSection.offsetTop - headerHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Active section highlighting
    window.addEventListener('scroll', function() {
        let current = '';
        const scrollPosition = window.scrollY + 100;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').substring(1) === current) {
                link.classList.add('active');
            }
        });
    });
}

// Chatbot functionality
function initializeChatbot() {
    const chatInput = document.getElementById('chatbot-input');
    const chatSend = document.getElementById('chatbot-send');
    const chatMessages = document.getElementById('chatbot-messages');

    const responses = {
        'experience': 'I have over 5 years of experience in Applied AI and Machine Learning, specializing in computer vision, NLP, and deep learning architectures. I\'ve worked on projects ranging from medical image analysis to real-time recommendation systems.',
        'skills': 'My technical expertise includes Python, TensorFlow, PyTorch, scikit-learn, OpenCV, and cloud platforms like AWS and GCP. I\'m also proficient in MLOps, data engineering, and model deployment at scale.',
        'projects': 'Some of my notable projects include developing a CNN-based medical diagnosis system with 95% accuracy, creating a real-time sentiment analysis tool for social media, and building an automated data pipeline that processes 1M+ records daily.',
        'music': 'Music is my creative outlet! I compose electronic and ambient pieces using AI-assisted tools. My compositions blend algorithmic generation with human creativity, exploring the intersection of technology and art.',
        'videos': 'I direct educational and artistic videos that make complex AI concepts accessible. My work includes documentaries on AI in healthcare, experimental films visualizing algorithms, and behind-the-scenes content about the creative process.',
        'education': 'I hold a Master\'s degree in Computer Science with a focus on Machine Learning. I\'m constantly learning through online courses, research papers, and hands-on projects. I believe in lifelong learning in this rapidly evolving field.',
        'collaboration': 'I\'m always open to interesting collaborations! Whether it\'s a challenging ML problem, a creative project, or educational content, I love working with passionate people who want to push boundaries.',
        'default': 'That\'s an interesting question! Could you be more specific? I can tell you about my technical experience, creative projects, education, or potential collaborations. What would you like to know more about?'
    };

    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-${isUser ? 'user' : 'robot'}"></i>
            </div>
            <div class="message-content">
                <p>${content}</p>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function getBotResponse(userInput) {
        const input = userInput.toLowerCase();
        
        // Simple keyword matching
        if (input.includes('experience') || input.includes('work') || input.includes('career')) {
            return responses.experience;
        } else if (input.includes('skill') || input.includes('technology') || input.includes('tool')) {
            return responses.skills;
        } else if (input.includes('project') || input.includes('portfolio') || input.includes('work')) {
            return responses.projects;
        } else if (input.includes('music') || input.includes('compose') || input.includes('song')) {
            return responses.music;
        } else if (input.includes('video') || input.includes('film') || input.includes('direct')) {
            return responses.videos;
        } else if (input.includes('education') || input.includes('study') || input.includes('learn')) {
            return responses.education;
        } else if (input.includes('collaborate') || input.includes('work together') || input.includes('hire')) {
            return responses.collaboration;
        } else {
            return responses.default;
        }
    }

    function sendMessage() {
        const userInput = chatInput.value.trim();
        if (!userInput) return;

        // Add user message
        addMessage(userInput, true);
        chatInput.value = '';

        // Simulate typing delay
        setTimeout(() => {
            const botResponse = getBotResponse(userInput);
            addMessage(botResponse);
        }, 1000);
    }

    chatSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
}

// Music player functionality
function initializeMusicPlayer() {
    const musicButtons = document.querySelectorAll('.btn-music');
    const generateButton = document.getElementById('generate-music');
    const compositionDisplay = document.getElementById('composition-display');
    const styleSelect = document.getElementById('music-style');
    const moodSelect = document.getElementById('music-mood');

    // Simulate music playback
    musicButtons.forEach(button => {
        button.addEventListener('click', function() {
            const trackId = this.getAttribute('data-track');
            const icon = this.querySelector('i');
            
            if (this.textContent.trim().startsWith('Play')) {
                // Start playing
                this.innerHTML = '<i class="fas fa-pause"></i> Pause';
                this.style.background = 'var(--gradient-secondary)';
                
                // Stop other tracks
                musicButtons.forEach(otherButton => {
                    if (otherButton !== this && otherButton.textContent.trim().startsWith('Pause')) {
                        otherButton.innerHTML = '<i class="fas fa-play"></i> Play';
                        otherButton.style.background = 'var(--gradient-primary)';
                    }
                });
                
                // Simulate track ending after some time
                setTimeout(() => {
                    if (this.textContent.trim().startsWith('Pause')) {
                        this.innerHTML = '<i class="fas fa-play"></i> Play';
                        this.style.background = 'var(--gradient-primary)';
                    }
                }, 5000); // 5 seconds for demo
                
            } else {
                // Stop playing
                this.innerHTML = '<i class="fas fa-play"></i> Play';
                this.style.background = 'var(--gradient-primary)';
            }
        });
    });

    // AI Music Generation
    if (generateButton) {
        generateButton.addEventListener('click', function() {
            const style = styleSelect.value;
            const mood = moodSelect.value;
            
            this.textContent = 'Generating...';
            this.disabled = true;
            
            compositionDisplay.innerHTML = '<div class="loading">🎵 AI is composing your music...</div>';
            
            setTimeout(() => {
                const composition = generateMusicComposition(style, mood);
                compositionDisplay.innerHTML = composition;
                this.textContent = 'Generate Music';
                this.disabled = false;
            }, 3000);
        });
    }

    function generateMusicComposition(style, mood) {
        const compositions = {
            ambient: {
                calm: `
🎹 Generated Ambient Composition - "Serene Waves"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

♪ Key: C Major
♫ Tempo: 60 BPM
🎵 Duration: 4:23

Structure:
┌─ Intro (0:00-0:45) ─ Gentle pad sweep, soft reverb
├─ Theme A (0:45-1:30) ─ Floating melody, minimal percussion
├─ Development (1:30-2:45) ─ Layered textures, ambient drones
├─ Theme B (2:45-3:30) ─ Ethereal bells, subtle bass movement
└─ Outro (3:30-4:23) ─ Fade to silence with sustained chords

Instruments: Synthesizer pads, soft piano, ambient textures
Mood: Peaceful, meditative, floating sensation
                `,
                energetic: `
🎹 Generated Ambient Composition - "Electric Dreams"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

♪ Key: A Minor
♫ Tempo: 85 BPM
🎵 Duration: 5:12

Structure:
┌─ Build-up (0:00-1:00) ─ Rising synth arpeggios
├─ Main Theme (1:00-2:30) ─ Pulsing bass, rhythmic pads
├─ Breakdown (2:30-3:15) ─ Filtered elements, space
├─ Climax (3:15-4:30) ─ Full arrangement, soaring lead
└─ Resolution (4:30-5:12) ─ Gentle fade with echoes

Instruments: Analog synthesizers, digital effects, subtle percussion
Mood: Uplifting, cosmic, forward-moving energy
                `
            },
            electronic: {
                calm: `
🎛️ Generated Electronic Composition - "Digital Zen"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

♪ Key: F Major
♫ Tempo: 70 BPM
🎵 Duration: 3:55

Structure:
┌─ Intro (0:00-0:30) ─ Soft filter sweeps, gentle clicks
├─ Verse (0:30-1:45) ─ Melodic synth lead, subtle beat
├─ Chorus (1:45-2:30) ─ Warm bass, lush pads
├─ Bridge (2:30-3:15) ─ Glitchy textures, space
└─ Outro (3:15-3:55) ─ Fade with vinyl crackle

Instruments: Digital synthesizers, soft drums, vocal chops
Mood: Relaxed, modern, contemplative
                `,
                energetic: `
🎛️ Generated Electronic Composition - "Neon Pulse"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

♪ Key: E Minor
♫ Tempo: 128 BPM
🎵 Duration: 4:44

Structure:
┌─ Intro (0:00-0:32) ─ Build-up with filter sweeps
├─ Drop (0:32-1:16) ─ Heavy bass, driving beat
├─ Breakdown (1:16-2:00) ─ Melodic interlude
├─ Second Drop (2:00-3:20) ─ Full energy, lead synth
└─ Outro (3:20-4:44) ─ Gradual wind-down

Instruments: Bass synthesizer, electronic drums, lead synth, effects
Mood: High-energy, danceable, futuristic
                `
            }
        };

        return compositions[style]?.[mood] || compositions.ambient.calm;
    }
}

// Video player functionality
function initializeVideoPlayer() {
    const videoPlayButtons = document.querySelectorAll('.video-play-btn');
    
    videoPlayButtons.forEach(button => {
        button.addEventListener('click', function() {
            const videoCard = this.closest('.video-card');
            const videoTitle = videoCard.querySelector('h3').textContent;
            
            // Create modal for video playback (simplified)
            const modal = document.createElement('div');
            modal.className = 'video-modal';
            modal.innerHTML = `
                <div class="video-modal-content">
                    <div class="video-modal-header">
                        <h3>${videoTitle}</h3>
                        <button class="video-modal-close">&times;</button>
                    </div>
                    <div class="video-placeholder-large">
                        <i class="fas fa-play-circle"></i>
                        <p>Video player would be embedded here</p>
                        <p><em>This is a demo - actual video files would be loaded here</em></p>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            // Close modal functionality
            const closeBtn = modal.querySelector('.video-modal-close');
            closeBtn.addEventListener('click', () => {
                document.body.removeChild(modal);
            });
            
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    document.body.removeChild(modal);
                }
            });
        });
    });
}

// Blog interactions
function initializeBlogInteractions() {
    const readMoreButtons = document.querySelectorAll('.blog-read-more');
    
    readMoreButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const blogCard = this.closest('.blog-card');
            const blogContent = blogCard.querySelector('.blog-content');
            
            if (blogContent) {
                if (blogContent.style.display === 'none' || !blogContent.style.display) {
                    blogContent.style.display = 'block';
                    this.textContent = 'Read Less';
                } else {
                    blogContent.style.display = 'none';
                    this.textContent = 'Read More';
                }
            }
        });
    });
}

// Scroll effects
function initializeScrollEffects() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for scroll animations
    const animatedElements = document.querySelectorAll('.blog-card, .music-card, .video-card');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// Mobile menu functionality
function initializeMobileMenu() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });

        // Close menu when clicking on a link
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
            });
        });
    }
}

// Additional CSS for modal and mobile menu
const additionalStyles = `
.video-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2000;
}

.video-modal-content {
    background: white;
    border-radius: var(--radius-lg);
    width: 90%;
    max-width: 800px;
    max-height: 90%;
    overflow: hidden;
}

.video-modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-lg);
    border-bottom: 1px solid var(--gray-light);
}

.video-modal-close {
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
    color: var(--gray-primary);
}

.video-placeholder-large {
    padding: var(--spacing-xl);
    text-align: center;
    color: var(--gray-primary);
}

.video-placeholder-large i {
    font-size: 4rem;
    margin-bottom: var(--spacing-md);
}

.loading {
    text-align: center;
    color: var(--primary-color);
    font-weight: 500;
    animation: pulse 1.5s infinite;
}

@media (max-width: 768px) {
    .nav-menu {
        position: fixed;
        top: 70px;
        left: -100%;
        width: 100%;
        height: calc(100vh - 70px);
        background: white;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        padding-top: var(--spacing-xl);
        transition: var(--transition-normal);
        box-shadow: var(--shadow-lg);
    }
    
    .nav-menu.active {
        left: 0;
    }
    
    .hamburger.active span:nth-child(1) {
        transform: rotate(-45deg) translate(-5px, 6px);
    }
    
    .hamburger.active span:nth-child(2) {
        opacity: 0;
    }
    
    .hamburger.active span:nth-child(3) {
        transform: rotate(45deg) translate(-5px, -6px);
    }
}
`;

// Inject additional styles
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

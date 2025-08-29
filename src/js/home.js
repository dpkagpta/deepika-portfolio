// Home Page Advanced Functionality

document.addEventListener('DOMContentLoaded', function() {
    initializePreloader();
    initializeParticles();
    initializeTypingEffect();
    initializeCounterAnimation();
    initializeSkillBars();
    initializeNeuralNetwork();
    initializeThemeToggle();
    initializeScrollEffects();
    initializeAOS();
});

// Preloader with neural network animation
function initializePreloader() {
    const preloader = document.getElementById('preloader');
    
    // Simulate loading time
    setTimeout(() => {
        preloader.classList.add('hidden');
        
        // Remove preloader from DOM after animation
        setTimeout(() => {
            if (preloader && preloader.parentNode) {
                preloader.parentNode.removeChild(preloader);
            }
        }, 500);
    }, 3000);
}

// Particles.js background
function initializeParticles() {
    if (typeof particlesJS !== 'undefined') {
        particlesJS('particles-js', {
            particles: {
                number: {
                    value: 50,
                    density: {
                        enable: true,
                        value_area: 800
                    }
                },
                color: {
                    value: ['#2563eb', '#7c3aed', '#059669']
                },
                shape: {
                    type: 'circle',
                    stroke: {
                        width: 0,
                        color: '#000000'
                    }
                },
                opacity: {
                    value: 0.3,
                    random: true,
                    anim: {
                        enable: true,
                        speed: 1,
                        opacity_min: 0.1,
                        sync: false
                    }
                },
                size: {
                    value: 3,
                    random: true,
                    anim: {
                        enable: true,
                        speed: 2,
                        size_min: 0.1,
                        sync: false
                    }
                },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#2563eb',
                    opacity: 0.2,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 1,
                    direction: 'none',
                    random: false,
                    straight: false,
                    out_mode: 'out',
                    bounce: false,
                    attract: {
                        enable: false,
                        rotateX: 600,
                        rotateY: 1200
                    }
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: {
                        enable: true,
                        mode: 'grab'
                    },
                    onclick: {
                        enable: true,
                        mode: 'push'
                    },
                    resize: true
                },
                modes: {
                    grab: {
                        distance: 140,
                        line_linked: {
                            opacity: 0.5
                        }
                    },
                    push: {
                        particles_nb: 4
                    }
                }
            },
            retina_detect: true
        });
    }
}

// Typing effect for hero title
function initializeTypingEffect() {
    const typingElement = document.getElementById('typing-text');
    if (!typingElement) return;
    
    const texts = [
        'AI Engineer',
        'Data Scientist',
        'ML Researcher',
        'Tech Innovator',
        'Creative Coder'
    ];
    
    let textIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    
    function typeEffect() {
        const currentText = texts[textIndex];
        
        if (isDeleting) {
            typingElement.textContent = currentText.substring(0, charIndex - 1);
            charIndex--;
        } else {
            typingElement.textContent = currentText.substring(0, charIndex + 1);
            charIndex++;
        }
        
        let typeSpeed = isDeleting ? 50 : 100;
        
        if (!isDeleting && charIndex === currentText.length) {
            typeSpeed = 2000; // Pause at end
            isDeleting = true;
        } else if (isDeleting && charIndex === 0) {
            isDeleting = false;
            textIndex = (textIndex + 1) % texts.length;
            typeSpeed = 500;
        }
        
        setTimeout(typeEffect, typeSpeed);
    }
    
    typeEffect();
}

// Animated counters for statistics
function initializeCounterAnimation() {
    const counters = document.querySelectorAll('.stat-number');
    
    const animateCounter = (counter) => {
        const target = parseInt(counter.getAttribute('data-count'));
        const increment = target / 50;
        let current = 0;
        
        const updateCounter = () => {
            if (current < target) {
                current += increment;
                counter.textContent = Math.ceil(current);
                requestAnimationFrame(updateCounter);
            } else {
                counter.textContent = target;
            }
        };
        
        updateCounter();
    };
    
    // Intersection Observer for counter animation
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                counterObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    counters.forEach(counter => {
        counterObserver.observe(counter);
    });
}

// Animated skill progress bars
function initializeSkillBars() {
    const skillBars = document.querySelectorAll('.skill-progress');
    
    const animateSkillBar = (bar) => {
        const width = bar.getAttribute('data-width');
        bar.style.width = width;
    };
    
    const skillObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateSkillBar(entry.target);
                skillObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    skillBars.forEach(bar => {
        skillObserver.observe(bar);
    });
}

// Interactive neural network visualization
function initializeNeuralNetwork() {
    const canvas = document.getElementById('neural-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const neurons = document.querySelectorAll('.neural-network-complex .neuron');
    
    // Set canvas size
    const resizeCanvas = () => {
        const container = canvas.parentElement;
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Get neuron positions
    const getNeuronPositions = () => {
        const positions = [];
        neurons.forEach(neuron => {
            const rect = neuron.getBoundingClientRect();
            const containerRect = canvas.getBoundingClientRect();
            positions.push({
                x: rect.left - containerRect.left + rect.width / 2,
                y: rect.top - containerRect.top + rect.height / 2
            });
        });
        return positions;
    };
    
    // Draw connections between neurons
    const drawConnections = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const positions = getNeuronPositions();
        
        // Input to hidden layer connections
        for (let i = 0; i < 3; i++) {
            for (let j = 3; j < 7; j++) {
                if (positions[i] && positions[j]) {
                    drawConnection(positions[i], positions[j], 0.3);
                }
            }
        }
        
        // Hidden to output layer connections
        for (let i = 3; i < 7; i++) {
            for (let j = 7; j < 9; j++) {
                if (positions[i] && positions[j]) {
                    drawConnection(positions[i], positions[j], 0.3);
                }
            }
        }
    };
    
    const drawConnection = (start, end, opacity) => {
        const gradient = ctx.createLinearGradient(start.x, start.y, end.x, end.y);
        gradient.addColorStop(0, `rgba(37, 99, 235, ${opacity})`);
        gradient.addColorStop(1, `rgba(124, 58, 237, ${opacity})`);
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
    };
    
    // Animate connections
    let animationPhase = 0;
    const animateConnections = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const positions = getNeuronPositions();
        
        // Pulsing opacity effect
        const pulseOpacity = 0.2 + 0.3 * Math.sin(animationPhase * 0.05);
        
        // Draw all connections with pulsing effect
        for (let i = 0; i < 3; i++) {
            for (let j = 3; j < 7; j++) {
                if (positions[i] && positions[j]) {
                    drawConnection(positions[i], positions[j], pulseOpacity);
                }
            }
        }
        
        for (let i = 3; i < 7; i++) {
            for (let j = 7; j < 9; j++) {
                if (positions[i] && positions[j]) {
                    drawConnection(positions[i], positions[j], pulseOpacity);
                }
            }
        }
        
        animationPhase++;
        requestAnimationFrame(animateConnections);
    };
    
    // Start animation
    animateConnections();
    
    // Add hover effects to neurons
    neurons.forEach((neuron, index) => {
        neuron.addEventListener('mouseenter', () => {
            neuron.style.transform = 'scale(1.3)';
            neuron.style.boxShadow = '0 8px 25px rgba(37, 99, 235, 0.6)';
        });
        
        neuron.addEventListener('mouseleave', () => {
            neuron.style.transform = 'scale(1)';
            neuron.style.boxShadow = '0 4px 15px rgba(37, 99, 235, 0.3)';
        });
        
        neuron.addEventListener('click', () => {
            // Create ripple effect
            const ripple = document.createElement('div');
            ripple.style.position = 'absolute';
            ripple.style.borderRadius = '50%';
            ripple.style.background = 'rgba(37, 99, 235, 0.3)';
            ripple.style.transform = 'scale(0)';
            ripple.style.animation = 'ripple 0.6s linear';
            ripple.style.left = '50%';
            ripple.style.top = '50%';
            ripple.style.width = '100px';
            ripple.style.height = '100px';
            ripple.style.marginLeft = '-50px';
            ripple.style.marginTop = '-50px';
            
            neuron.style.position = 'relative';
            neuron.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // Add ripple animation CSS
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
}

// Theme toggle functionality
function initializeThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) return;
    
    const body = document.body;
    const icon = themeToggle.querySelector('i');
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        body.classList.add('dark-theme');
        icon.classList.replace('fa-moon', 'fa-sun');
    }
    
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('dark-theme');
        
        if (body.classList.contains('dark-theme')) {
            icon.classList.replace('fa-moon', 'fa-sun');
            localStorage.setItem('theme', 'dark');
        } else {
            icon.classList.replace('fa-sun', 'fa-moon');
            localStorage.setItem('theme', 'light');
        }
    });
}

// Advanced scroll effects
function initializeScrollEffects() {
    // Parallax effect for hero section
    const hero = document.querySelector('.hero');
    const heroContent = document.querySelector('.hero-content');
    
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.5;
        
        if (hero) {
            hero.style.transform = `translateY(${rate}px)`;
        }
    });
    
    // Scroll progress indicator
    const createScrollProgress = () => {
        const progress = document.createElement('div');
        progress.id = 'scroll-progress';
        progress.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 3px;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            z-index: 9999;
            transition: width 0.1s ease;
        `;
        document.body.appendChild(progress);
        
        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset;
            const docHeight = document.body.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;
            progress.style.width = scrollPercent + '%';
        });
    };
    
    createScrollProgress();
    
    // Smooth scroll for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Initialize AOS (Animate On Scroll)
function initializeAOS() {
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 1000,
            once: true,
            offset: 100,
            easing: 'ease-out-cubic'
        });
    }
}

// Enhanced portfolio card interactions
document.addEventListener('DOMContentLoaded', function() {
    const portfolioCards = document.querySelectorAll('.portfolio-card');
    
    portfolioCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            // Add glowing effect
            this.style.boxShadow = '0 20px 60px rgba(37, 99, 235, 0.2)';
            
            // Animate icon
            const icon = this.querySelector('.card-icon');
            if (icon) {
                icon.style.transform = 'scale(1.1) rotate(5deg)';
            }
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.boxShadow = '0 10px 40px rgba(0, 0, 0, 0.1)';
            
            const icon = this.querySelector('.card-icon');
            if (icon) {
                icon.style.transform = 'scale(1) rotate(0deg)';
            }
        });
    });
});

// Easter egg: Konami code activation
function initializeEasterEgg() {
    const konamiCode = [
        'ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown',
        'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight',
        'KeyB', 'KeyA'
    ];
    
    let userInput = [];
    
    document.addEventListener('keydown', function(e) {
        userInput.push(e.code);
        userInput = userInput.slice(-konamiCode.length);
        
        if (userInput.join('') === konamiCode.join('')) {
            activateEasterEgg();
        }
    });
    
    function activateEasterEgg() {
        // Add rainbow animation to the entire page
        document.body.style.animation = 'rainbow 2s infinite';
        
        // Show success message
        const message = document.createElement('div');
        message.textContent = '🎉 Easter Egg Activated! Welcome to the Matrix! 🎉';
        message.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
            color: white;
            padding: 20px;
            border-radius: 15px;
            font-weight: bold;
            z-index: 10000;
            animation: bounce 1s infinite;
        `;
        
        document.body.appendChild(message);
        
        setTimeout(() => {
            document.body.style.animation = '';
            message.remove();
        }, 5000);
    }
    
    // Add rainbow animation CSS
    const style = document.createElement('style');
    style.textContent = `
        @keyframes rainbow {
            0% { filter: hue-rotate(0deg); }
            100% { filter: hue-rotate(360deg); }
        }
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translate(-50%, -50%) translateY(0); }
            40% { transform: translate(-50%, -50%) translateY(-10px); }
            60% { transform: translate(-50%, -50%) translateY(-5px); }
        }
    `;
    document.head.appendChild(style);
}

// Initialize easter egg
initializeEasterEgg();

// Performance monitoring
function initializePerformanceMonitoring() {
    if ('performance' in window) {
        window.addEventListener('load', () => {
            const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
            console.log(`🚀 Page loaded in ${loadTime}ms`);
            
            // Log performance metrics
            const metrics = {
                loadTime: loadTime,
                domContentLoaded: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
                firstPaint: performance.getEntriesByType('paint')[0]?.startTime || 0
            };
            
            console.log('📊 Performance Metrics:', metrics);
        });
    }
}

initializePerformanceMonitoring();

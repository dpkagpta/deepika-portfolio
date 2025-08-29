// About Page Functionality

document.addEventListener('DOMContentLoaded', function() {
    initializeSkillBars();
    initializeContactForm();
    initializeNetworkDiagram();
    initializeScrollAnimations();
    initializeAOS();
});

// Skill bars animation
function initializeSkillBars() {
    const skillBars = document.querySelectorAll('.level-bar');
    
    const animateSkillBar = (bar) => {
        const width = bar.getAttribute('style').match(/--width:\s*(\d+%)/);
        if (width) {
            bar.style.width = width[1];
            bar.setAttribute('data-width', width[1]);
        }
    };
    
    const skillObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    animateSkillBar(entry.target);
                }, 200);
                skillObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    skillBars.forEach(bar => {
        // Extract width from CSS custom property
        const computedStyle = getComputedStyle(bar);
        const width = bar.style.getPropertyValue('--width') || '0%';
        bar.style.width = '0%';
        bar.style.setProperty('--width', width);
        
        skillObserver.observe(bar);
    });
}

// Contact form functionality
function initializeContactForm() {
    const form = document.getElementById('contact-form');
    
    if (!form) return;
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);
        
        // Validate form
        if (!validateForm(data)) {
            return;
        }
        
        // Simulate form submission
        submitContactForm(data);
    });
}

function validateForm(data) {
    const errors = [];
    
    if (!data.name || data.name.trim().length < 2) {
        errors.push('Name must be at least 2 characters long');
    }
    
    if (!data.email || !isValidEmail(data.email)) {
        errors.push('Please enter a valid email address');
    }
    
    if (!data.subject) {
        errors.push('Please select a subject');
    }
    
    if (!data.message || data.message.trim().length < 10) {
        errors.push('Message must be at least 10 characters long');
    }
    
    if (errors.length > 0) {
        showNotification(errors.join('\n'), 'error');
        return false;
    }
    
    return true;
}

function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function submitContactForm(data) {
    const submitBtn = document.querySelector('.submit-btn');
    const originalText = submitBtn.innerHTML;
    
    // Show loading state
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
    submitBtn.disabled = true;
    
    // Simulate API call
    setTimeout(() => {
        // Success
        submitBtn.innerHTML = '<i class="fas fa-check"></i> Message Sent!';
        submitBtn.style.background = '#10b981';
        
        showNotification('Thank you for your message! I\'ll get back to you soon.', 'success');
        
        // Reset form
        document.getElementById('contact-form').reset();
        
        // Reset button after delay
        setTimeout(() => {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
            submitBtn.style.background = '';
        }, 3000);
        
    }, 2000);
}

// Network diagram animation
function initializeNetworkDiagram() {
    const connections = document.querySelector('.connections');
    if (!connections) return;
    
    // Set up SVG viewBox
    connections.setAttribute('viewBox', '0 0 300 300');
    
    // Animate connections on scroll
    const diagramObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateNetworkConnections();
                diagramObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    const diagram = document.querySelector('.network-diagram');
    if (diagram) {
        diagramObserver.observe(diagram);
    }
}

function animateNetworkConnections() {
    const lines = document.querySelectorAll('.connections line');
    
    lines.forEach((line, index) => {
        line.style.strokeDasharray = '1000';
        line.style.strokeDashoffset = '1000';
        line.style.animation = `drawLine 2s ease-out ${index * 0.3}s forwards`;
    });
    
    // Add CSS for line drawing animation
    if (!document.querySelector('#line-animation-style')) {
        const style = document.createElement('style');
        style.id = 'line-animation-style';
        style.textContent = `
            @keyframes drawLine {
                to {
                    stroke-dashoffset: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// Scroll animations for timeline and other elements
function initializeScrollAnimations() {
    const timelineItems = document.querySelectorAll('.timeline-item');
    const publicationItems = document.querySelectorAll('.publication-item');
    const awardItems = document.querySelectorAll('.award-item');
    
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
    
    // Initialize elements for animation
    [...timelineItems, ...publicationItems, ...awardItems].forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
        observer.observe(el);
    });
    
    // Stagger timeline animations
    timelineItems.forEach((item, index) => {
        item.style.transitionDelay = `${index * 0.2}s`;
    });
    
    // Stagger award animations
    awardItems.forEach((item, index) => {
        item.style.transitionDelay = `${index * 0.1}s`;
    });
}

// Interactive hover effects
function initializeInteractiveEffects() {
    // Timeline item interactions
    const timelineItems = document.querySelectorAll('.timeline-item');
    timelineItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.querySelector('.timeline-marker').style.transform = 'scale(1.2)';
            this.querySelector('.timeline-marker').style.boxShadow = '0 0 0 6px white, 0 0 0 12px rgba(37, 99, 235, 0.3)';
        });
        
        item.addEventListener('mouseleave', function() {
            this.querySelector('.timeline-marker').style.transform = 'scale(1)';
            this.querySelector('.timeline-marker').style.boxShadow = '0 0 0 6px white, 0 0 0 8px rgba(37, 99, 235, 0.2)';
        });
    });
    
    // Publication item interactions
    const publicationItems = document.querySelectorAll('.publication-item');
    publicationItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.borderLeftWidth = '8px';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.borderLeftWidth = '4px';
        });
    });
    
    // Network node interactions
    const nodes = document.querySelectorAll('.node');
    nodes.forEach(node => {
        node.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.2)';
            this.style.zIndex = '10';
        });
        
        node.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.zIndex = '1';
        });
    });
    
    // Central node interaction
    const centralNode = document.querySelector('.central-node');
    if (centralNode) {
        centralNode.addEventListener('mouseenter', function() {
            this.style.transform = 'translate(-50%, -50%) scale(1.1)';
            this.style.boxShadow = '0 15px 40px rgba(37, 99, 235, 0.4)';
        });
        
        centralNode.addEventListener('mouseleave', function() {
            this.style.transform = 'translate(-50%, -50%) scale(1)';
            this.style.boxShadow = '0 10px 30px rgba(37, 99, 235, 0.3)';
        });
    }
}

// Contact method interactions
function initializeContactMethods() {
    const contactMethods = document.querySelectorAll('.contact-method');
    
    contactMethods.forEach(method => {
        method.addEventListener('click', function(e) {
            const icon = this.querySelector('i');
            
            if (icon.classList.contains('fa-envelope')) {
                // Email
                e.preventDefault();
                const email = 'dr.deepi@ai-portfolio.com';
                if (navigator.clipboard) {
                    navigator.clipboard.writeText(email);
                    showNotification('Email address copied to clipboard!', 'success');
                } else {
                    window.location.href = `mailto:${email}`;
                }
            } else if (icon.classList.contains('fa-linkedin') || 
                      icon.classList.contains('fa-github') || 
                      icon.classList.contains('fa-twitter')) {
                // Social links
                e.preventDefault();
                showNotification('Social profile link would open here', 'info');
            }
        });
    });
}

// Form field enhancements
function initializeFormEnhancements() {
    const formInputs = document.querySelectorAll('.form-group input, .form-group select, .form-group textarea');
    
    formInputs.forEach(input => {
        // Add floating label effect
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            if (!this.value) {
                this.parentElement.classList.remove('focused');
            }
        });
        
        // Check if already has value
        if (input.value) {
            input.parentElement.classList.add('focused');
        }
    });
    
    // Character count for textarea
    const textarea = document.querySelector('textarea[name="message"]');
    if (textarea) {
        const charCount = document.createElement('div');
        charCount.className = 'char-count';
        charCount.style.cssText = `
            text-align: right;
            font-size: 0.8rem;
            color: var(--gray-primary);
            margin-top: 0.5rem;
        `;
        textarea.parentElement.appendChild(charCount);
        
        textarea.addEventListener('input', function() {
            const count = this.value.length;
            charCount.textContent = `${count} characters`;
            
            if (count < 10) {
                charCount.style.color = '#ef4444';
            } else if (count > 500) {
                charCount.style.color = '#f59e0b';
            } else {
                charCount.style.color = 'var(--gray-primary)';
            }
        });
        
        // Initial count
        textarea.dispatchEvent(new Event('input'));
    }
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
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
        max-width: 300px;
        white-space: pre-line;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, type === 'error' ? 5000 : 3000);
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

// Initialize all interactive effects
document.addEventListener('DOMContentLoaded', function() {
    initializeInteractiveEffects();
    initializeContactMethods();
    initializeFormEnhancements();
});

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
    
    .form-group.focused label {
        transform: translateY(-10px) scale(0.9);
        color: var(--primary-color);
    }
    
    .form-group label {
        transition: all 0.3s ease;
        transform-origin: left top;
    }
`;
document.head.appendChild(style);

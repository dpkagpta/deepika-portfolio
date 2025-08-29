// Videos Page Functionality

document.addEventListener('DOMContentLoaded', function() {
    initializeVideoFilters();
    initializeVideoPlayers();
    initializeVideoModal();
    initializeFeaturedVideo();
    initializeLoadMore();
    initializeVideoInteractions();
    initializeAOS();
});

// Video category filters
function initializeVideoFilters() {
    const filterBtns = document.querySelectorAll('.filter-btn');
    const videoCards = document.querySelectorAll('.video-card');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const category = this.getAttribute('data-category');
            
            // Update active button
            filterBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            // Filter video cards
            filterVideoCards(category, videoCards);
        });
    });
}

function filterVideoCards(category, cards) {
    cards.forEach(card => {
        const cardCategory = card.getAttribute('data-category');
        
        if (category === 'all' || cardCategory === category) {
            card.style.display = 'block';
            card.style.animation = 'fadeInUp 0.5s ease-out';
        } else {
            card.style.display = 'none';
        }
    });
}

// Video player functionality
function initializeVideoPlayers() {
    const playButtons = document.querySelectorAll('.play-btn');
    
    playButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const videoId = this.getAttribute('data-video');
            openVideoModal(videoId);
        });
    });
}

// Video modal functionality
function initializeVideoModal() {
    const modal = document.getElementById('video-modal');
    const modalClose = document.getElementById('modal-close');
    const modalTitle = document.getElementById('modal-video-title');
    const modalContent = document.getElementById('modal-video-content');
    const modalDescription = document.getElementById('modal-video-description');

    if (modalClose) {
        modalClose.addEventListener('click', closeVideoModal);
    }

    if (modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeVideoModal();
            }
        });
    }

    // ESC key to close modal
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal && modal.classList.contains('active')) {
            closeVideoModal();
        }
    });
}

function openVideoModal(videoId) {
    const modal = document.getElementById('video-modal');
    const modalTitle = document.getElementById('modal-video-title');
    const modalContent = document.getElementById('modal-video-content');
    const modalDescription = document.getElementById('modal-video-description');
    
    if (!modal) return;

    // Video content database
    const videos = {
        'medical-ai': {
            title: 'Medical AI: Diagnosing the Future',
            content: `
                <div style="background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; padding: 4rem; text-align: center; border-radius: 1rem;">
                    <i class="fas fa-microscope" style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.8;"></i>
                    <h3 style="margin-bottom: 1rem;">Medical AI Research Documentary</h3>
                    <p style="opacity: 0.9;">An in-depth exploration of AI applications in medical diagnosis</p>
                    <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 0.5rem;">
                        <p><i class="fas fa-info-circle"></i> This is a demo video player</p>
                        <p>Actual video content would be embedded here</p>
                    </div>
                </div>
            `,
            description: 'This documentary examines how machine learning algorithms are revolutionizing medical diagnosis, featuring interviews with leading researchers and real-world case studies from hospitals implementing AI-powered diagnostic systems.'
        },
        'neural-tutorial': {
            title: 'Building Neural Networks from Scratch',
            content: `
                <div style="background: linear-gradient(135deg, #059669, #2563eb); color: white; padding: 4rem; text-align: center; border-radius: 1rem;">
                    <i class="fas fa-code" style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.8;"></i>
                    <h3 style="margin-bottom: 1rem;">Neural Network Tutorial</h3>
                    <p style="opacity: 0.9;">Complete implementation guide with Python</p>
                    <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 0.5rem;">
                        <p><i class="fas fa-play-circle"></i> 22:18 Tutorial</p>
                        <p>Step-by-step coding walkthrough</p>
                    </div>
                </div>
            `,
            description: 'A comprehensive tutorial covering the fundamentals of neural networks, from basic perceptrons to complex architectures. Includes practical Python implementation without using high-level frameworks, perfect for understanding the underlying mathematics.'
        },
        'ai-art': {
            title: 'When Algorithms Dream: AI Art',
            content: `
                <div style="background: linear-gradient(135deg, #f59e0b, #dc2626); color: white; padding: 4rem; text-align: center; border-radius: 1rem;">
                    <i class="fas fa-palette" style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.8;"></i>
                    <h3 style="margin-bottom: 1rem;">AI Art Generation</h3>
                    <p style="opacity: 0.9;">Exploring creativity through algorithms</p>
                    <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 0.5rem;">
                        <p><i class="fas fa-magic"></i> Experimental Film</p>
                        <p>8:45 artistic exploration</p>
                    </div>
                </div>
            `,
            description: 'An experimental short film that visualizes the creative process of AI art generation, featuring style transfer techniques, GANs, and the philosophical implications of machine creativity.'
        },
        'cv-demo': {
            title: 'Real-time Computer Vision Systems',
            content: `
                <div style="background: linear-gradient(135deg, #8b5cf6, #06b6d4); color: white; padding: 4rem; text-align: center; border-radius: 1rem;">
                    <i class="fas fa-robot" style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.8;"></i>
                    <h3 style="margin-bottom: 1rem;">Computer Vision Demo</h3>
                    <p style="opacity: 0.9;">Real-time object detection and tracking</p>
                    <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 0.5rem;">
                        <p><i class="fas fa-eye"></i> Live Demonstration</p>
                        <p>11:22 technical showcase</p>
                    </div>
                </div>
            `,
            description: 'Live demonstrations of advanced computer vision applications including real-time object detection, facial recognition, pose estimation, and autonomous navigation systems running on edge devices.'
        },
        'creative-process': {
            title: 'My Creative Workflow: Tech Meets Art',
            content: `
                <div style="background: linear-gradient(135deg, #ec4899, #f59e0b); color: white; padding: 4rem; text-align: center; border-radius: 1rem;">
                    <i class="fas fa-video" style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.8;"></i>
                    <h3 style="margin-bottom: 1rem;">Creative Process</h3>
                    <p style="opacity: 0.9;">Behind the scenes of creative technology</p>
                    <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 0.5rem;">
                        <p><i class="fas fa-camera"></i> Behind the Scenes</p>
                        <p>6:33 personal insight</p>
                    </div>
                </div>
            `,
            description: 'A personal look into my creative workflow, showing how I balance technical rigor with artistic expression, from initial concept to final production in both my video and music projects.'
        },
        'ai-ethics': {
            title: 'AI Ethics: Navigating the Future',
            content: `
                <div style="background: linear-gradient(135deg, #1e293b, #475569); color: white; padding: 4rem; text-align: center; border-radius: 1rem;">
                    <i class="fas fa-globe" style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.8;"></i>
                    <h3 style="margin-bottom: 1rem;">AI Ethics Exploration</h3>
                    <p style="opacity: 0.9;">The moral implications of artificial intelligence</p>
                    <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.1); border-radius: 0.5rem;">
                        <p><i class="fas fa-balance-scale"></i> Documentary</p>
                        <p>18:15 ethical discussion</p>
                    </div>
                </div>
            `,
            description: 'A thought-provoking documentary examining the ethical implications of AI development, featuring discussions with ethicists, researchers, and industry leaders about responsible AI deployment.'
        }
    };

    const video = videos[videoId] || {
        title: 'Video Not Found',
        content: '<p>The requested video could not be found.</p>',
        description: 'Video description not available.'
    };

    modalTitle.textContent = video.title;
    modalContent.innerHTML = video.content;
    modalDescription.innerHTML = `<p>${video.description}</p>`;
    
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeVideoModal() {
    const modal = document.getElementById('video-modal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

// Featured video functionality
function initializeFeaturedVideo() {
    const featuredPlayBtn = document.getElementById('featured-play');
    const controlButtons = document.querySelectorAll('.control-btn');
    
    if (featuredPlayBtn) {
        featuredPlayBtn.addEventListener('click', function() {
            // Simulate video playback
            simulateFeaturedPlayback();
        });
    }

    controlButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const icon = this.querySelector('i');
            
            if (icon.classList.contains('fa-play')) {
                icon.classList.replace('fa-play', 'fa-pause');
                simulateFeaturedPlayback();
            } else if (icon.classList.contains('fa-pause')) {
                icon.classList.replace('fa-pause', 'fa-play');
                stopFeaturedPlayback();
            } else if (icon.classList.contains('fa-volume-up')) {
                icon.classList.replace('fa-volume-up', 'fa-volume-mute');
            } else if (icon.classList.contains('fa-volume-mute')) {
                icon.classList.replace('fa-volume-mute', 'fa-volume-up');
            } else if (icon.classList.contains('fa-expand')) {
                enterFullscreen();
            }
        });
    });
}

function simulateFeaturedPlayback() {
    const progressBar = document.querySelector('.featured-video .progress');
    const timeDisplay = document.querySelector('.time-display');
    
    if (!progressBar) return;

    let currentTime = 0;
    const duration = 765; // 12:45 in seconds
    
    const interval = setInterval(() => {
        currentTime += 1;
        const progress = (currentTime / duration) * 100;
        progressBar.style.width = progress + '%';
        
        if (timeDisplay) {
            const current = formatTime(currentTime);
            const total = formatTime(duration);
            timeDisplay.textContent = `${current} / ${total}`;
        }
        
        if (currentTime >= duration) {
            clearInterval(interval);
            resetFeaturedPlayer();
        }
    }, 100); // Fast demo playback
}

function stopFeaturedPlayback() {
    const progressBar = document.querySelector('.featured-video .progress');
    const timeDisplay = document.querySelector('.time-display');
    
    if (progressBar) {
        progressBar.style.width = '25%'; // Reset to original position
    }
    
    if (timeDisplay) {
        timeDisplay.textContent = '0:00 / 12:45';
    }
}

function resetFeaturedPlayer() {
    const playBtn = document.querySelector('.featured-video .control-btn');
    if (playBtn) {
        const icon = playBtn.querySelector('i');
        icon.classList.replace('fa-pause', 'fa-play');
    }
    
    stopFeaturedPlayback();
}

function enterFullscreen() {
    const videoPlayer = document.querySelector('.video-player');
    if (videoPlayer && videoPlayer.requestFullscreen) {
        videoPlayer.requestFullscreen();
    }
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Load more videos functionality
function initializeLoadMore() {
    const loadMoreBtn = document.getElementById('load-more-videos');
    
    if (!loadMoreBtn) return;

    let loadedCount = 6; // Initially loaded videos
    const totalCount = 20; // Total available videos

    loadMoreBtn.addEventListener('click', function() {
        const button = this;
        const originalText = button.innerHTML;

        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading Videos...';
        button.disabled = true;

        // Simulate loading delay
        setTimeout(() => {
            const newVideos = createMoreVideos(3);
            const videosGrid = document.querySelector('.videos-grid');
            
            newVideos.forEach((video, index) => {
                videosGrid.appendChild(video);
                // Animate in
                setTimeout(() => {
                    video.style.opacity = '1';
                    video.style.transform = 'translateY(0)';
                }, index * 100);
            });

            loadedCount += 3;

            if (loadedCount >= totalCount) {
                button.style.display = 'none';
            } else {
                button.innerHTML = originalText;
                button.disabled = false;
            }
        }, 1500);
    });
}

function createMoreVideos(count) {
    const videoTemplates = [
        {
            category: 'tutorial',
            title: 'Advanced PyTorch Techniques',
            excerpt: 'Deep dive into advanced PyTorch features for production ML systems.',
            duration: '28:45',
            views: '8.1K'
        },
        {
            category: 'ai-showcase',
            title: 'Generative AI in Action',
            excerpt: 'Live demonstration of text-to-image and text-to-video generation models.',
            duration: '15:30',
            views: '12.4K'
        },
        {
            category: 'documentary',
            title: 'The Future of Work and AI',
            excerpt: 'Exploring how artificial intelligence will reshape the job market.',
            duration: '24:12',
            views: '6.8K'
        }
    ];

    const videos = [];

    for (let i = 0; i < count && i < videoTemplates.length; i++) {
        const template = videoTemplates[i];
        const video = document.createElement('div');
        video.className = 'video-card';
        video.setAttribute('data-category', template.category);
        video.style.opacity = '0';
        video.style.transform = 'translateY(20px)';
        video.style.transition = 'all 0.5s ease';
        
        video.innerHTML = `
            <div class="video-thumbnail">
                <div class="thumbnail-placeholder ${template.category}">
                    <i class="fas fa-${template.category === 'tutorial' ? 'chalkboard-teacher' : template.category === 'ai-showcase' ? 'magic' : 'film'}"></i>
                    <div class="overlay-text">${template.title}</div>
                </div>
                <div class="video-duration">${template.duration}</div>
                <div class="play-overlay">
                    <button class="play-btn" data-video="new-video-${i}">
                        <i class="fas fa-play"></i>
                    </button>
                </div>
            </div>
            <div class="video-content">
                <h3>${template.title}</h3>
                <p class="video-excerpt">${template.excerpt}</p>
                <div class="video-metadata">
                    <span class="category ${template.category}">${template.category.replace('-', ' ')}</span>
                    <span class="views">${template.views} views</span>
                    <span class="date">Dec 2024</span>
                </div>
            </div>
        `;
        
        // Add click handler for new video
        const playBtn = video.querySelector('.play-btn');
        playBtn.addEventListener('click', function() {
            openVideoModal('new-video-demo');
        });
        
        videos.push(video);
    }
    
    return videos;
}

// Video interaction enhancements
function initializeVideoInteractions() {
    const videoCards = document.querySelectorAll('.video-card');
    const actionBtns = document.querySelectorAll('.action-btn');
    const seriesBtns = document.querySelectorAll('.series-btn');

    // Video card hover effects
    videoCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Action button interactions
    actionBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const action = this.classList.contains('like-btn') ? 'liked' :
                          this.classList.contains('share-btn') ? 'shared' : 'saved';
            
            if (action === 'liked') {
                const span = this.querySelector('span');
                const count = parseInt(span.textContent) + 1;
                span.textContent = count;
                this.style.background = '#ef4444';
                this.style.color = 'white';
            }
            
            showNotification(`Video ${action} successfully!`, 'success');
        });
    });

    // Series button interactions
    seriesBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            showNotification('Opening video series...', 'info');
        });
    });
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
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

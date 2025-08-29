// Blog Page Functionality

document.addEventListener('DOMContentLoaded', function() {
    initializeBlogSearch();
    initializeBlogFilters();
    initializeFeaturedVisualization();
    initializeBlogGrid();
    initializeArticleModal();
    initializeNewsletterForm();
    initializeViewToggle();
    initializeLoadMore();
    initializeAOS();
});

// Blog search functionality
function initializeBlogSearch() {
    const searchInput = document.getElementById('blog-search');
    const searchClear = document.getElementById('search-clear');
    const blogCards = document.querySelectorAll('.blog-card');

    if (!searchInput) return;

    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        
        // Show/hide clear button
        if (searchTerm) {
            searchClear.classList.add('visible');
        } else {
            searchClear.classList.remove('visible');
        }

        // Filter blog cards
        filterBlogCards();
    });

    searchClear.addEventListener('click', function() {
        searchInput.value = '';
        this.classList.remove('visible');
        filterBlogCards();
    });

    // Auto-complete suggestions
    const suggestions = [
        'machine learning', 'deep learning', 'neural networks', 'computer vision',
        'natural language processing', 'data science', 'AI research', 'transformers',
        'CNN', 'RNN', 'LSTM', 'GAN', 'reinforcement learning', 'supervised learning',
        'unsupervised learning', 'feature engineering', 'model deployment'
    ];

    searchInput.addEventListener('focus', function() {
        // Could implement autocomplete dropdown here
    });
}

// Blog filters functionality
function initializeBlogFilters() {
    const categoryFilter = document.getElementById('category-filter');
    const sortFilter = document.getElementById('sort-filter');
    const filterReset = document.getElementById('filter-reset');

    if (!categoryFilter || !sortFilter) return;

    categoryFilter.addEventListener('change', filterBlogCards);
    sortFilter.addEventListener('change', filterBlogCards);

    filterReset.addEventListener('click', function() {
        categoryFilter.value = 'all';
        sortFilter.value = 'newest';
        document.getElementById('blog-search').value = '';
        document.getElementById('search-clear').classList.remove('visible');
        filterBlogCards();
    });
}

function filterBlogCards() {
    const searchTerm = document.getElementById('blog-search').value.toLowerCase();
    const categoryFilter = document.getElementById('category-filter').value;
    const sortFilter = document.getElementById('sort-filter').value;
    const blogCards = Array.from(document.querySelectorAll('.blog-card'));

    // Filter cards
    let filteredCards = blogCards.filter(card => {
        const title = card.querySelector('.card-title').textContent.toLowerCase();
        const excerpt = card.querySelector('.card-excerpt').textContent.toLowerCase();
        const category = card.getAttribute('data-category');
        const tags = Array.from(card.querySelectorAll('.tag')).map(tag => tag.textContent.toLowerCase());

        // Search filter
        const matchesSearch = !searchTerm || 
            title.includes(searchTerm) || 
            excerpt.includes(searchTerm) ||
            tags.some(tag => tag.includes(searchTerm));

        // Category filter
        const matchesCategory = categoryFilter === 'all' || category === categoryFilter;

        return matchesSearch && matchesCategory;
    });

    // Sort cards
    filteredCards.sort((a, b) => {
        switch (sortFilter) {
            case 'newest':
                return new Date(b.getAttribute('data-date')) - new Date(a.getAttribute('data-date'));
            case 'oldest':
                return new Date(a.getAttribute('data-date')) - new Date(b.getAttribute('data-date'));
            case 'popular':
                return parseInt(b.getAttribute('data-popularity')) - parseInt(a.getAttribute('data-popularity'));
            case 'reading-time':
                const aTime = parseInt(a.querySelector('.read-time').textContent);
                const bTime = parseInt(b.querySelector('.read-time').textContent);
                return aTime - bTime;
            default:
                return 0;
        }
    });

    // Hide all cards first
    blogCards.forEach(card => {
        card.style.display = 'none';
        card.style.animation = '';
    });

    // Show filtered cards with animation
    setTimeout(() => {
        filteredCards.forEach((card, index) => {
            card.style.display = 'block';
            card.style.animation = `fadeInUp 0.5s ease-out ${index * 0.1}s both`;
        });
    }, 100);

    // Update results count
    updateResultsCount(filteredCards.length, blogCards.length);
}

function updateResultsCount(filtered, total) {
    // You could add a results counter here
    console.log(`Showing ${filtered} of ${total} articles`);
}

// Featured article data visualization
function initializeFeaturedVisualization() {
    const canvas = document.getElementById('data-connections');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dataPoints = document.querySelectorAll('.data-point');
    
    // Set canvas size
    const resizeCanvas = () => {
        const container = canvas.parentElement;
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Get data point positions
    const getDataPointPositions = () => {
        const positions = [];
        dataPoints.forEach(point => {
            const rect = point.getBoundingClientRect();
            const containerRect = canvas.getBoundingClientRect();
            positions.push({
                x: rect.left - containerRect.left + rect.width / 2,
                y: rect.top - containerRect.top + rect.height / 2
            });
        });
        return positions;
    };

    // Draw connections between data points
    const drawConnections = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const positions = getDataPointPositions();
        
        // Draw connections between nearby points
        for (let i = 0; i < positions.length; i++) {
            for (let j = i + 1; j < positions.length; j++) {
                const distance = Math.sqrt(
                    Math.pow(positions[i].x - positions[j].x, 2) + 
                    Math.pow(positions[i].y - positions[j].y, 2)
                );
                
                if (distance < 150) { // Only connect nearby points
                    drawConnection(positions[i], positions[j], 1 - distance / 150);
                }
            }
        }
    };

    const drawConnection = (start, end, opacity) => {
        const gradient = ctx.createLinearGradient(start.x, start.y, end.x, end.y);
        gradient.addColorStop(0, `rgba(37, 99, 235, ${opacity * 0.6})`);
        gradient.addColorStop(1, `rgba(124, 58, 237, ${opacity * 0.6})`);

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
        drawConnections();
        animationPhase += 0.02;
        requestAnimationFrame(animateConnections);
    };

    // Start animation
    setTimeout(animateConnections, 1000); // Delay to ensure data points are positioned
}

// Blog grid interactions
function initializeBlogGrid() {
    const blogCards = document.querySelectorAll('.blog-card');
    
    blogCards.forEach(card => {
        // Hover effects
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
            
            // Animate card image
            const cardImage = this.querySelector('.card-image');
            if (cardImage) {
                cardImage.style.transform = 'scale(1.1)';
            }
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
            
            const cardImage = this.querySelector('.card-image');
            if (cardImage) {
                cardImage.style.transform = 'scale(1)';
            }
        });

        // Read more functionality
        const readMoreBtn = card.querySelector('.read-more');
        if (readMoreBtn) {
            readMoreBtn.addEventListener('click', function() {
                const articleId = this.getAttribute('data-article');
                openArticleModal(articleId);
            });
        }
    });
}

// Article modal functionality
function initializeArticleModal() {
    const modal = document.getElementById('article-modal');
    const modalClose = document.getElementById('modal-close');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');

    // Featured article read button
    const featuredReadBtn = document.querySelector('.read-article');
    if (featuredReadBtn) {
        featuredReadBtn.addEventListener('click', function() {
            const articleId = this.getAttribute('data-article');
            openArticleModal(articleId);
        });
    }

    if (modalClose) {
        modalClose.addEventListener('click', closeArticleModal);
    }

    if (modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeArticleModal();
            }
        });
    }

    // ESC key to close modal
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal && modal.classList.contains('active')) {
            closeArticleModal();
        }
    });
}

function openArticleModal(articleId) {
    const modal = document.getElementById('article-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    
    if (!modal) return;

    // Article content database
    const articles = {
        'dataset-curation': {
            title: 'The Art and Science of Dataset Curation in Modern AI',
            content: `
                <div class="article-content">
                    <h3>Understanding Dataset Quality</h3>
                    <p>High-quality datasets are the foundation of successful machine learning projects. They must be representative, diverse, and free from systematic biases that could skew model performance.</p>
                    
                    <h3>Collection Strategies</h3>
                    <ul>
                        <li><strong>Stratified Sampling:</strong> Ensuring balanced representation across different categories</li>
                        <li><strong>Temporal Considerations:</strong> Accounting for time-based variations in data</li>
                        <li><strong>Cross-Domain Validation:</strong> Testing dataset applicability across different contexts</li>
                    </ul>
                    
                    <h3>Bias Mitigation Techniques</h3>
                    <p>Addressing bias requires a multi-pronged approach:</p>
                    <ol>
                        <li>Statistical analysis of demographic distributions</li>
                        <li>Adversarial testing for hidden biases</li>
                        <li>Continuous monitoring and adjustment protocols</li>
                    </ol>
                    
                    <h3>Tools and Technologies</h3>
                    <p>Modern dataset curation leverages powerful tools like Apache Spark for large-scale processing, TensorFlow Data Validation for quality checks, and custom Python pipelines for automated preprocessing.</p>
                    
                    <pre><code class="language-python">
import tensorflow_data_validation as tfdv
from apache_beam.options.pipeline_options import PipelineOptions

# Generate statistics for training data
train_stats = tfdv.generate_statistics_from_csv(
    data_location='train_data.csv'
)

# Infer schema from statistics
schema = tfdv.infer_schema(statistics=train_stats)

# Validate new data against schema
validation_stats = tfdv.generate_statistics_from_csv(
    data_location='validation_data.csv'
)
anomalies = tfdv.validate_statistics(
    statistics=validation_stats, 
    schema=schema
)
                    </code></pre>
                    
                    <h3>Future Directions</h3>
                    <p>As we move towards more sophisticated AI systems, dataset curation will increasingly involve synthetic data generation, federated learning approaches, and privacy-preserving techniques that maintain data utility while protecting individual privacy.</p>
                </div>
            `
        },
        'transformers': {
            title: 'Transformer Architectures: Beyond Attention',
            content: `
                <div class="article-content">
                    <h3>The Evolution of Transformers</h3>
                    <p>Since the introduction of the "Attention is All You Need" paper, transformer architectures have revolutionized not just NLP, but the entire field of machine learning.</p>
                    
                    <h3>Core Mechanisms</h3>
                    <p>The self-attention mechanism allows models to weigh the importance of different words in a sequence when processing each word:</p>
                    
                    <pre><code class="language-python">
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attention_output)
                    </code></pre>
                    
                    <h3>Beyond Language: Vision Transformers</h3>
                    <p>Vision Transformers (ViTs) have shown that the transformer architecture can be successfully applied to computer vision tasks, often outperforming traditional CNNs.</p>
                    
                    <h3>Future Innovations</h3>
                    <p>Recent developments include sparse attention mechanisms, memory-efficient transformers, and multimodal architectures that can process text, images, and audio simultaneously.</p>
                </div>
            `
        }
        // Add more articles as needed
    };

    const article = articles[articleId] || {
        title: 'Article Not Found',
        content: '<p>The requested article could not be found.</p>'
    };

    modalTitle.textContent = article.title;
    modalBody.innerHTML = article.content;
    
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Initialize syntax highlighting if Prism is available
    if (typeof Prism !== 'undefined') {
        Prism.highlightAll();
    }
}

function closeArticleModal() {
    const modal = document.getElementById('article-modal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

// Newsletter form functionality
function initializeNewsletterForm() {
    const form = document.getElementById('newsletter-form');
    if (!form) return;

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const email = this.querySelector('input[type="email"]').value;
        const button = this.querySelector('button');
        const originalText = button.innerHTML;

        // Simulate subscription process
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Subscribing...';
        button.disabled = true;

        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-check"></i> Subscribed!';
            button.style.background = '#10b981';
            
            // Show success message
            showNotification('Successfully subscribed to newsletter!', 'success');
            
            // Reset form
            setTimeout(() => {
                this.querySelector('input[type="email"]').value = '';
                button.innerHTML = originalText;
                button.disabled = false;
                button.style.background = '';
            }, 2000);
        }, 2000);
    });
}

// View toggle functionality
function initializeViewToggle() {
    const viewToggles = document.querySelectorAll('.view-toggle');
    const blogGrid = document.getElementById('blog-grid');

    viewToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const view = this.getAttribute('data-view');
            
            // Update active state
            viewToggles.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // Update grid view
            if (view === 'list') {
                blogGrid.classList.add('list-view');
            } else {
                blogGrid.classList.remove('list-view');
            }
        });
    });
}

// Load more functionality
function initializeLoadMore() {
    const loadMoreBtn = document.getElementById('load-more');
    if (!loadMoreBtn) return;

    let loadedCount = 6; // Initially loaded articles
    const totalCount = 24; // Total available articles

    loadMoreBtn.addEventListener('click', function() {
        const button = this;
        const originalText = button.innerHTML;

        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
        button.disabled = true;

        // Simulate loading delay
        setTimeout(() => {
            // Create new article cards (simplified)
            const newArticles = createMoreArticles(3);
            const blogGrid = document.getElementById('blog-grid');
            
            newArticles.forEach((article, index) => {
                blogGrid.appendChild(article);
                // Animate in
                setTimeout(() => {
                    article.style.opacity = '1';
                    article.style.transform = 'translateY(0)';
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

function createMoreArticles(count) {
    // This would typically fetch from an API
    const articles = [];
    const templates = [
        {
            category: 'machine-learning',
            title: 'Advanced Feature Engineering Techniques',
            excerpt: 'Exploring sophisticated methods for creating meaningful features from raw data.',
            readTime: '7 min'
        },
        {
            category: 'ai-research',
            title: 'Quantum Machine Learning: Current State and Future',
            excerpt: 'An overview of quantum computing applications in machine learning.',
            readTime: '14 min'
        },
        {
            category: 'tutorials',
            title: 'Building Scalable ML APIs with FastAPI',
            excerpt: 'Step-by-step guide to creating production-ready machine learning APIs.',
            readTime: '20 min'
        }
    ];

    for (let i = 0; i < count && i < templates.length; i++) {
        const template = templates[i];
        const article = document.createElement('article');
        article.className = 'blog-card';
        article.setAttribute('data-category', template.category);
        article.setAttribute('data-date', '2024-11-01');
        article.setAttribute('data-popularity', '75');
        article.style.opacity = '0';
        article.style.transform = 'translateY(20px)';
        article.style.transition = 'all 0.5s ease';
        
        article.innerHTML = `
            <div class="card-image">
                <div class="neural-network-mini">
                    <div class="mini-node"></div>
                    <div class="mini-node"></div>
                    <div class="mini-node"></div>
                </div>
            </div>
            <div class="card-content">
                <div class="article-meta">
                    <span class="category ${template.category}">${template.category.replace('-', ' ')}</span>
                    <span class="read-time"><i class="fas fa-clock"></i> ${template.readTime}</span>
                </div>
                <h3 class="card-title">${template.title}</h3>
                <p class="card-excerpt">${template.excerpt}</p>
                <div class="card-footer">
                    <div class="article-tags">
                        <span class="tag">New</span>
                    </div>
                    <button class="read-more">
                        <span>Read More</span>
                        <i class="fas fa-arrow-right"></i>
                    </button>
                </div>
            </div>
        `;
        
        articles.push(article);
    }
    
    return articles;
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'success' ? 'check' : 'info'}-circle"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : '#2563eb'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        z-index: 3000;
        animation: slideInRight 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Initialize AOS
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
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
`;
document.head.appendChild(style);

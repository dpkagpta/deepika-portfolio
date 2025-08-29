// Music Page Functionality

document.addEventListener('DOMContentLoaded', function() {
    initializeAudioVisualizer();
    initializeAIComposer();
    initializeMusicLibrary();
    initializeSequencer();
    initializeWaveforms();
    initializeAOS();
});

// Audio Visualizer
function initializeAudioVisualizer() {
    const canvas = document.getElementById('visualizer-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let animationId;

    const resizeCanvas = () => {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Create audio visualization
    const visualize = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = Math.min(centerX, centerY) * 0.8;
        
        // Draw circular audio visualization
        const numBars = 64;
        const angleStep = (Math.PI * 2) / numBars;
        
        for (let i = 0; i < numBars; i++) {
            const angle = i * angleStep;
            const barHeight = 20 + Math.sin(Date.now() * 0.005 + i * 0.2) * 30;
            
            const x1 = centerX + Math.cos(angle) * radius;
            const y1 = centerY + Math.sin(angle) * radius;
            const x2 = centerX + Math.cos(angle) * (radius + barHeight);
            const y2 = centerY + Math.sin(angle) * (radius + barHeight);
            
            const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
            gradient.addColorStop(0, '#2563eb');
            gradient.addColorStop(1, '#7c3aed');
            
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 3;
            ctx.lineCap = 'round';
            
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
        
        animationId = requestAnimationFrame(visualize);
    };

    visualize();

    // Cleanup
    return () => {
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
    };
}

// AI Composer
function initializeAIComposer() {
    const generateBtn = document.getElementById('generate-music');
    const compositionDisplay = document.getElementById('composition-display');
    const outputControls = document.getElementById('output-controls');
    const tempoSlider = document.getElementById('music-tempo');
    const tempoValue = document.getElementById('tempo-value');

    if (!generateBtn) return;

    // Tempo slider
    if (tempoSlider && tempoValue) {
        tempoSlider.addEventListener('input', function() {
            tempoValue.textContent = this.value;
        });
    }

    generateBtn.addEventListener('click', function() {
        const genre = document.getElementById('music-genre').value;
        const mood = document.getElementById('music-mood').value;
        const tempo = document.getElementById('music-tempo').value;
        const length = document.getElementById('music-length').value;

        // Disable button during generation
        this.disabled = true;
        this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

        // Simulate AI composition generation
        setTimeout(() => {
            const composition = generateComposition(genre, mood, tempo, length);
            displayComposition(composition);
            
            // Show controls
            if (outputControls) {
                outputControls.style.display = 'flex';
            }

            // Reset button
            this.disabled = false;
            this.innerHTML = '<i class="fas fa-magic"></i> Generate Composition';
        }, 3000);
    });

    // Playback controls
    const playBtn = document.getElementById('play-composition');
    const downloadBtn = document.getElementById('download-composition');
    const shareBtn = document.getElementById('share-composition');

    if (playBtn) {
        playBtn.addEventListener('click', function() {
            const icon = this.querySelector('i');
            if (icon.classList.contains('fa-play')) {
                icon.classList.replace('fa-play', 'fa-pause');
                simulatePlayback();
            } else {
                icon.classList.replace('fa-pause', 'fa-play');
                stopPlayback();
            }
        });
    }

    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            showNotification('Composition download started!', 'success');
        });
    }

    if (shareBtn) {
        shareBtn.addEventListener('click', function() {
            if (navigator.share) {
                navigator.share({
                    title: 'AI Generated Music',
                    text: 'Check out this AI-generated composition!',
                    url: window.location.href
                });
            } else {
                showNotification('Share link copied to clipboard!', 'success');
            }
        });
    }
}

function generateComposition(genre, mood, tempo, length) {
    const compositions = {
        ambient: {
            calm: `🎹 AI Composition Generated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Genre: Ambient • Mood: Calm • Tempo: ${tempo} BPM
Duration: ${Math.floor(length / 60)}:${(length % 60).toString().padStart(2, '0')}

🎵 Structure Analysis:
┌─ Intro (0:00-0:20) ─ Soft pad layers, gentle reverb
├─ Development (0:20-1:40) ─ Floating melodies, ambient textures
├─ Peak (1:40-2:20) ─ Harmonic convergence, ethereal bells
└─ Outro (2:20-${Math.floor(length / 60)}:${(length % 60).toString().padStart(2, '0')}) ─ Fade to silence

🔊 Instrumentation:
• Synthesizer pads (40Hz-2kHz)
• Ambient textures (white noise filtered)
• Bell-like tones (sine waves, delay)
• Sub-bass (60Hz fundamental)

📊 Harmonic Analysis:
Key: C Major | Scale: Natural Minor
Chord Progression: Cmaj7 - Am7 - Fmaj7 - G7sus4
Tempo Variations: ±5 BPM (organic feel)

🤖 AI Parameters:
Model: TransformerMusic-v2.1
Training Data: 10K ambient compositions
Creativity Factor: 0.7 (balanced innovation)
Harmonic Complexity: Medium`,
            energetic: `🎹 AI Composition Generated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Genre: Ambient • Mood: Energetic • Tempo: ${tempo} BPM
Duration: ${Math.floor(length / 60)}:${(length % 60).toString().padStart(2, '0')}

🎵 Structure Analysis:
┌─ Build-up (0:00-0:30) ─ Rising arpeggios, filtered pads
├─ Main Section (0:30-2:00) ─ Pulsing rhythms, evolving textures
├─ Climax (2:00-2:30) ─ Full harmonic spectrum, dynamic peaks
└─ Resolution (2:30-${Math.floor(length / 60)}:${(length % 60).toString().padStart(2, '0')}) ─ Gradual energy dissipation

🔊 Instrumentation:
• Arpeggiated synthesizers
• Rhythmic pad sequences
• Dynamic filtering effects
• Percussive elements (subtle)

📊 Harmonic Analysis:
Key: E Minor | Mode: Dorian
Chord Progression: Em - G - D - C - Am - B7
Energy Curve: Exponential rise to linear decay

🤖 AI Parameters:
Model: EnergeticAmbient-GAN
Innovation Level: High (0.8)
Rhythmic Complexity: Medium-High`
        },
        electronic: {
            energetic: `🎛️ AI Composition Generated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Genre: Electronic • Mood: Energetic • Tempo: ${tempo} BPM
Duration: ${Math.floor(length / 60)}:${(length % 60).toString().padStart(2, '0')}

🎵 Structure Analysis:
┌─ Intro (0:00-0:15) ─ Filter sweeps, building tension
├─ Drop (0:15-1:00) ─ Heavy bass, driving drums
├─ Breakdown (1:00-1:30) ─ Melodic interlude
├─ Second Drop (1:30-2:30) ─ Full arrangement, peak energy
└─ Outro (2:30-${Math.floor(length / 60)}:${(length % 60).toString().padStart(2, '0')}) ─ Filtered wind-down

🔊 Sound Design:
• Sub bass (50-80Hz) - Monophonic, side-chained
• Lead synth (1-4kHz) - Saw wave, filtered
• Drums - 909-style kicks, crisp snares
• FX - Reverb, delay, filtering automation

📊 Technical Specs:
BPM: ${tempo} | Key: A Minor
Compression: 4:1 ratio, fast attack
EQ: Bass boost @60Hz, presence @3kHz
Limiting: -0.1dB peak, -14 LUFS integrated

🤖 AI Generation:
Model: ElectroBeats-Transformer
Dataset: 50K electronic tracks
Style Transfer: Applied from reference tracks
Mastering: AI-automated loudness optimization`
        }
    };

    const genreCompositions = compositions[genre] || compositions.ambient;
    return genreCompositions[mood] || genreCompositions.calm;
}

function displayComposition(composition) {
    const display = document.getElementById('composition-display');
    if (display) {
        display.innerHTML = `<pre style="text-align: left; white-space: pre-wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; line-height: 1.4;">${composition}</pre>`;
    }
}

function simulatePlayback() {
    const progress = document.getElementById('music-progress');
    if (!progress) return;

    let width = 0;
    const interval = setInterval(() => {
        width += 1;
        progress.style.width = width + '%';
        
        if (width >= 100) {
            clearInterval(interval);
            // Reset play button
            const playBtn = document.getElementById('play-composition');
            if (playBtn) {
                const icon = playBtn.querySelector('i');
                icon.classList.replace('fa-pause', 'fa-play');
            }
            progress.style.width = '0%';
        }
    }, 100);
}

function stopPlayback() {
    const progress = document.getElementById('music-progress');
    if (progress) {
        progress.style.width = '0%';
    }
}

// Music Library
function initializeMusicLibrary() {
    const filterBtns = document.querySelectorAll('.filter-btn');
    const musicCards = document.querySelectorAll('.music-card');
    const playButtons = document.querySelectorAll('.play-button');

    // Filter functionality
    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const filter = this.getAttribute('data-filter');
            
            // Update active button
            filterBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            // Filter cards
            musicCards.forEach(card => {
                const genre = card.getAttribute('data-genre');
                if (filter === 'all' || genre === filter) {
                    card.style.display = 'block';
                    card.style.animation = 'fadeInUp 0.5s ease-out';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });

    // Play button functionality
    playButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const trackId = this.getAttribute('data-track');
            const icon = this.querySelector('i');
            
            // Stop all other tracks
            playButtons.forEach(otherBtn => {
                if (otherBtn !== this) {
                    const otherIcon = otherBtn.querySelector('i');
                    otherIcon.classList.replace('fa-pause', 'fa-play');
                }
            });
            
            // Toggle current track
            if (icon.classList.contains('fa-play')) {
                icon.classList.replace('fa-play', 'fa-pause');
                simulateTrackPlayback(trackId);
            } else {
                icon.classList.replace('fa-pause', 'fa-play');
            }
        });
    });
}

function simulateTrackPlayback(trackId) {
    // Simulate track playback (in a real app, this would control actual audio)
    setTimeout(() => {
        const btn = document.querySelector(`[data-track="${trackId}"]`);
        if (btn) {
            const icon = btn.querySelector('i');
            icon.classList.replace('fa-pause', 'fa-play');
        }
    }, 5000); // 5 seconds for demo
}

// Interactive Sequencer
function initializeSequencer() {
    const playBtn = document.getElementById('sequencer-play');
    const stopBtn = document.getElementById('sequencer-stop');
    const clearBtn = document.getElementById('sequencer-clear');
    const bpmInput = document.getElementById('sequencer-bpm');
    const stepGrid = document.getElementById('step-grid');

    if (!stepGrid) return;

    let isPlaying = false;
    let currentStep = 0;
    let sequenceInterval;
    let bpm = 120;

    // Create step grid
    createStepGrid();

    function createStepGrid() {
        const tracks = ['kick', 'snare', 'hihat', 'synth'];
        const steps = 16;

        stepGrid.innerHTML = '';
        stepGrid.style.gridTemplateRows = `repeat(${tracks.length}, 1fr)`;

        tracks.forEach((track, trackIndex) => {
            for (let step = 0; step < steps; step++) {
                const stepElement = document.createElement('div');
                stepElement.className = 'step';
                stepElement.setAttribute('data-track', track);
                stepElement.setAttribute('data-step', step);
                
                stepElement.addEventListener('click', function() {
                    this.classList.toggle('active');
                });

                stepGrid.appendChild(stepElement);
            }
        });
    }

    // Playback controls
    if (playBtn) {
        playBtn.addEventListener('click', function() {
            if (!isPlaying) {
                startSequence();
                this.querySelector('i').classList.replace('fa-play', 'fa-pause');
            } else {
                pauseSequence();
                this.querySelector('i').classList.replace('fa-pause', 'fa-play');
            }
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', function() {
            stopSequence();
            if (playBtn) {
                playBtn.querySelector('i').classList.replace('fa-pause', 'fa-play');
            }
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            const steps = stepGrid.querySelectorAll('.step');
            steps.forEach(step => {
                step.classList.remove('active', 'playing');
            });
        });
    }

    if (bpmInput) {
        bpmInput.addEventListener('change', function() {
            bpm = parseInt(this.value);
            if (isPlaying) {
                stopSequence();
                startSequence();
            }
        });
    }

    function startSequence() {
        isPlaying = true;
        currentStep = 0;
        const stepDuration = (60 / bpm / 4) * 1000; // 16th notes

        sequenceInterval = setInterval(() => {
            playStep(currentStep);
            currentStep = (currentStep + 1) % 16;
        }, stepDuration);
    }

    function pauseSequence() {
        isPlaying = false;
        if (sequenceInterval) {
            clearInterval(sequenceInterval);
        }
        clearStepHighlights();
    }

    function stopSequence() {
        isPlaying = false;
        currentStep = 0;
        if (sequenceInterval) {
            clearInterval(sequenceInterval);
        }
        clearStepHighlights();
    }

    function playStep(step) {
        // Clear previous highlights
        clearStepHighlights();

        // Highlight current step
        const currentSteps = stepGrid.querySelectorAll(`[data-step="${step}"]`);
        currentSteps.forEach(stepElement => {
            stepElement.classList.add('playing');
            
            // Trigger sound for active steps
            if (stepElement.classList.contains('active')) {
                const track = stepElement.getAttribute('data-track');
                triggerSound(track);
            }
        });
    }

    function clearStepHighlights() {
        const steps = stepGrid.querySelectorAll('.step');
        steps.forEach(step => {
            step.classList.remove('playing');
        });
    }

    function triggerSound(track) {
        // In a real implementation, this would trigger actual audio samples
        console.log(`Playing ${track} sound`);
        
        // Visual feedback
        showNotification(`🎵 ${track.toUpperCase()}`, 'info');
    }
}

// Waveform visualization
function initializeWaveforms() {
    const waveformCanvases = document.querySelectorAll('.waveform-canvas');
    
    waveformCanvases.forEach(canvas => {
        const ctx = canvas.getContext('2d');
        drawWaveform(ctx, canvas.width, canvas.height);
    });
}

function drawWaveform(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);
    
    const centerY = height / 2;
    const points = 100;
    const amplitude = height * 0.3;
    
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    
    ctx.beginPath();
    for (let i = 0; i < points; i++) {
        const x = (i / points) * width;
        const frequency = 0.02 + Math.random() * 0.01;
        const y = centerY + Math.sin(i * frequency) * amplitude * (Math.random() * 0.5 + 0.5);
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
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
    }, 2000);
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

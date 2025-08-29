#!/usr/bin/env python3
"""
Production optimization script for Deepika Gupta's portfolio website
Optimizes CSS, JS, and HTML for production deployment
"""

import os
import re
import json
from pathlib import Path

def minify_css(content):
    """Basic CSS minification"""
    # Remove comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content)
    # Remove whitespace around certain characters
    content = re.sub(r'\s*([{}:;,>+~])\s*', r'\1', content)
    return content.strip()

def minify_js(content):
    """Basic JS minification"""
    # Remove single-line comments
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    # Remove multi-line comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content)
    return content.strip()

def optimize_html(content):
    """Basic HTML optimization"""
    # Remove comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    # Remove extra whitespace between tags
    content = re.sub(r'>\s+<', '><', content)
    return content.strip()

def create_production_build():
    """Create optimized production build"""
    print("🚀 Creating production build...")
    
    # Create dist directory
    dist_dir = Path('dist')
    dist_dir.mkdir(exist_ok=True)
    
    # Copy and optimize files
    src_dir = Path('.')
    
    for file_path in src_dir.rglob('*'):
        if file_path.is_file() and not any(skip in str(file_path) for skip in ['.git', 'dist', '__pycache__', '.py']):
            relative_path = file_path.relative_to(src_dir)
            dest_path = dist_dir / relative_path
            
            # Create directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Optimize based on file type
                if file_path.suffix == '.css':
                    content = minify_css(content)
                    print(f"✅ Optimized CSS: {relative_path}")
                elif file_path.suffix == '.js':
                    content = minify_js(content)
                    print(f"✅ Optimized JS: {relative_path}")
                elif file_path.suffix == '.html':
                    content = optimize_html(content)
                    print(f"✅ Optimized HTML: {relative_path}")
                
                with open(dest_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            except UnicodeDecodeError:
                # Binary files - just copy
                with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())
                print(f"📁 Copied: {relative_path}")
    
    print(f"\n🎉 Production build created in 'dist' directory!")

def create_manifest():
    """Create web app manifest for PWA features"""
    manifest = {
        "name": "Deepika Gupta - AI/ML Engineer Portfolio",
        "short_name": "Deepika Portfolio",
        "description": "Professional portfolio of Deepika Gupta, Applied AI/ML Engineer",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#2563eb",
        "icons": [
            {
                "src": "/src/assets/icons/icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/src/assets/icons/icon-512.png", 
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }
    
    with open('manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("✅ Created PWA manifest.json")

if __name__ == "__main__":
    print("🔧 Optimizing Deepika Gupta's Portfolio for Production")
    print("=" * 50)
    
    create_manifest()
    create_production_build()
    
    print("\n🚀 Ready for deployment!")
    print("Next steps:")
    print("1. Run deploy.bat to push to production")
    print("2. Configure your hosting platform")
    print("3. Set up custom domain")

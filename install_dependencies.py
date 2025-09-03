#!/usr/bin/env python3
"""
Install required dependencies for the point cloud completion system
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install all required dependencies"""
    print("Installing required dependencies for Point Cloud Completion...")
    print("=" * 60)
    
    packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "shapely>=1.7.0"
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"Installation completed: {success_count}/{total_count} packages installed successfully")
    
    if success_count == total_count:
        print("✓ All dependencies installed successfully!")
        print("You can now run the point cloud completion system.")
    else:
        print("✗ Some dependencies failed to install.")
        print("Please install them manually using pip.")
    
    print("\nTo test the installation, run:")
    print("python test_conversion.py")

if __name__ == "__main__":
    main()

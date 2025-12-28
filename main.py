"""
Main Entry Point for Age Prediction System

This is the main entry point for running the biometric age prediction application.
Run this file to start the interactive age prediction system.
"""

import sys

# Add src directory to path
sys.path.insert(0, 'src')
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


from app.age_prediction import AgePrediction


def main():
    """Main entry point for the age prediction application."""
    program = AgePrediction()
    program.program_interface()


if __name__ == "__main__":
    main()

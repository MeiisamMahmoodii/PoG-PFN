#!/bin/bash
# Quick start script for PoG-PFN

echo "=================================="
echo "PoG-PFN Quick Start"
echo "=================================="

# Check Python version
echo -e "\n1. Checking Python version..."
python3 --version

# Create virtual environment
echo -e "\n2. Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "\n3. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run tests
echo -e "\n4. Running component tests..."
echo "  - Testing SCM Generator..."
python3 -m pog_pfn.data.scm_generator

echo "  - Testing Claim Generator..."
python3 -m pog_pfn.data.claim_generator

echo "  - Testing Dataset Encoder..."
python3 -m pog_pfn.models.dataset_encoder

echo "  - Testing Claim Encoder..."
python3 -m pog_pfn.models.claim_encoder

echo "  - Testing Graph Posterior..."
python3 -m pog_pfn.models.graph_posterior

echo "  - Testing Identification..."
python3 -m pog_pfn.models.identification

echo "  - Testing Effect Estimator..."
python3 -m pog_pfn.models.effect_estimator

echo "  - Testing Full Model..."
python3 -m pog_pfn.models.pog_pfn

# Run training
echo -e "\n5. Running training demo..."
python3 scripts/train.py

echo -e "\n=================================="
echo "Setup Complete!"
echo "=================================="

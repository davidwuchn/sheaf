#!/bin/bash

if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found. Installing..."
    pip install streamlit
    echo ""
fi

echo "Starting dashboard at http://localhost:8501"
streamlit run visualizer.py

#!/bin/bash

if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found."
    echo "Install with: pip install streamlit matplotlib numpy"
    exit 1
fi

echo "Starting dashboard at http://localhost:8501"
streamlit run visualizer.py

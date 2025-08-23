# Dockerfile for AI Trading System
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add new dependencies for AI integrations and dashboard
RUN pip install google-generativeai openai azure-identity azure-ai-openai streamlit pandas numpy

# Create directories for each agent
RUN mkdir -p /app/data /app/logs /app/models

# Environment variables for Upstox API
ENV UPSTOX_API_KEY=""
ENV UPSTOX_API_SECRET=""

# Expose port for monitoring dashboard
EXPOSE 8000

CMD ["python", "-m", "main"]
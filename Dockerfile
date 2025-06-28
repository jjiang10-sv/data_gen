# Multi-stage build for combined frontend + backend
FROM node:18-alpine AS frontend-builder

WORKDIR /app

# Copy package files
COPY web/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend code and build
COPY web/ ./

# Clean up unnecessary files
RUN rm -rf api/ || true
RUN rm -rf storage/ || true
RUN rm -rf .git/ || true
RUN rm -rf .next/ || true
RUN rm -rf .local/ || true

# Build frontend
RUN npm run build

# Backend stage
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for combined container
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app

# Copy pyproject.toml and poetry.lock
COPY pyproject.toml poetry.lock ./

# Install Poetry and basic dependencies (skip heavy ML packages for testing)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --only=main --no-root || pip install fastapi uvicorn python-dotenv pydantic

# Copy starfish source code and README (needed by backend)
COPY src/ ./src/
COPY README.md ./

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/.next ./web/.next
COPY --from=frontend-builder /app/public ./web/public
COPY --from=frontend-builder /app/package.json ./web/package.json
#COPY --from=frontend-builder /app/node_modules ./web/node_modules

# Copy backend API code
COPY web/api/ ./web/api/

# Copy configuration files
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY nginx.conf /etc/nginx/nginx.conf

# Create necessary directories and set permissions
RUN mkdir -p /var/log/supervisor /var/log/nginx /var/run \
    && chmod +x /app/src/ || true

# Expose port 7860 (required for Hugging Face Spaces)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start supervisor which manages both nginx and the applications
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"] 
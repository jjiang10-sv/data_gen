# Hugging Face Spaces Deployment

This guide explains how to deploy your combined FastAPI backend and Next.js frontend to Hugging Face Spaces.

> ✅ **Build Status**: Docker build is working successfully with resolved path alias issues!

## Overview

The `Dockerfile.huggingface` creates a single container that runs:
- **FastAPI backend** on port 8002
- **Next.js frontend** on port 3000  
- **Nginx reverse proxy** on port 7860 (required by Hugging Face Spaces)
- **Supervisor** to manage all processes

## Files for Hugging Face Spaces

1. **`Dockerfile`** - Combined Dockerfile for both services (multi-stage build)
2. **`nginx.conf`** - Nginx configuration for routing
3. **`supervisord.conf`** - Process manager configuration
4. **`.dockerignore`** - Optimized to exclude only necessary files
5. **`next.config.js`** - Enhanced with webpack path alias configuration
6. **`tsconfig.json`** - Updated with explicit path mappings

## Deployment Steps

### 1. Prepare Your Repository

Your repository is already configured with the correct `Dockerfile` for Hugging Face Spaces deployment.

### 2. Set Environment Variables in Hugging Face Spaces

In your Hugging Face Space settings, add these secrets:
- `GOOGLE_API_KEY` - Your Google API key
- `OPENAI_API_KEY` - Your OpenAI API key

### 3. Configure Your Space

- **Space Type**: Docker
- **Visibility**: Public or Private (your choice)
- **Hardware**: CPU Basic (or upgrade if needed)

### 4. Update API URLs in Frontend

Make sure your frontend points to the correct API endpoints:
```typescript
// In your frontend code, use relative URLs:
const API_BASE_URL = "/api"  // This goes to Next.js API routes in src/app/api/

// Next.js API routes will then proxy to FastAPI using:
// SERVER_BASE_URL=http://localhost:8002 (set in Dockerfile)
```

### 5. Deploy

1. Push your code to the Hugging Face Space repository
2. The space will automatically build and deploy

## How It Works

### Architecture
```
External Request :7860
        ↓
    Nginx Proxy
        ↓
    Next.js :3000 (handles ALL routes)
        ↓
    /api/* → src/app/api/ routes
        ↓
    proxy.ts uses SERVER_BASE_URL
        ↓
    FastAPI Backend :8002
```

### Port Mapping
- **7860** - Main port (required by Hugging Face Spaces)
- **3000** - Next.js frontend (internal) - handles all routing
- **8002** - FastAPI backend (internal) - accessed via Next.js proxy

### URL Routing
- `/` - Next.js frontend (all routes handled by Next.js)
- `/api/*` - Next.js API routes (in `src/app/api/`) that proxy to FastAPI backend
- `/backend-docs` - Direct FastAPI documentation (for debugging)
- `/backend-openapi.json` - Direct FastAPI OpenAPI schema (for debugging)

### Process Management
Supervisor manages three processes:
1. **backend** - FastAPI server (port 8002)
2. **frontend** - Next.js server (port 3000) - handles all routing and proxying
3. **nginx** - Reverse proxy (port 7860) - routes all traffic to Next.js

## Troubleshooting

### Common Issues

1. **Build fails with "Module not found: Can't resolve '@/lib/utils'"**
   - **FIXED**: This was caused by `lib/` being excluded in `.dockerignore`
   - The issue has been resolved by removing the `lib/` exclusion pattern

2. **Build fails during npm install**
   - Check that all package.json dependencies are valid
   - Ensure Node.js version compatibility

3. **FastAPI fails to start**
   - Check environment variables are set
   - Verify the starfish package is properly configured
   - Check logs in the Space's logs tab

4. **Frontend can't reach backend**
   - Ensure API calls use relative URLs (`/api/...`)
   - Check that `SERVER_BASE_URL=http://localhost:8002` is set in the Dockerfile
   - Verify Next.js API routes in `src/app/api/` are proxying correctly
   - For direct FastAPI access, use `/backend-docs` instead of `/docs`

5. **Space shows "Application starting" indefinitely**
   - Check supervisor logs for errors
   - Verify all services are starting properly

### Viewing Logs

In your Hugging Face Space:
1. Go to the "Logs" tab
2. Look for errors from supervisor, nginx, backend, or frontend
3. Logs are also written to `/var/log/` in the container

### Local Testing

Test the Hugging Face build locally:
```bash
# Build the image
docker build -t starfishai-web  .

# Run with environment variables
docker run -p 7860:7860 3000:3000 8002:8002\
  -e GOOGLE_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  starfishai-web 
```

Then visit:
- http://localhost:7860 - Main application
- http://localhost:7860/backend-docs - Direct FastAPI documentation
- http://localhost:7860/backend-openapi.json - Direct FastAPI schema

## Recent Fixes & Improvements

### Path Alias Resolution Fixed
- **Issue**: Build was failing with `Module not found: Can't resolve '@/lib/utils'`
- **Root Cause**: The `.dockerignore` file was excluding the `lib/` directory
- **Solution**: Removed `lib/` from `.dockerignore` and enhanced path configuration
- **Files Updated**: 
  - `.dockerignore` - Removed generic `lib/` exclusion
  - `next.config.js` - Added explicit webpack path aliases
  - `tsconfig.json` - Enhanced path mappings

### Docker Build Optimization
- **Multi-stage build** for optimal image size
- **Specific Python exclusions** in `.dockerignore` (e.g., `api/__pycache__/` instead of all `__pycache__/`)
- **Enhanced file copying strategy** during build

## Performance Tips

1. **Use CPU Basic** for development, upgrade for production
2. **Optimize Docker image** by removing unnecessary files
3. **Use caching** for build dependencies
4. **Monitor resource usage** in the Space dashboard

## Security Notes

- Never commit API keys to your repository
- Use Hugging Face Spaces secrets for sensitive environment variables
- Consider making your Space private if it contains sensitive data
- Regularly update dependencies for security patches 

docker run -d -p 7860:7860 --name starfish-app -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf -v $(pwd)/supervisord.conf:/etc/supervisor/conf.d/supervisord.conf starfish-app

docker build -t starfish-app . 
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

To **completely remove Git LFS** from your repo (i.e. make all files tracked by LFS become normal Git files again), follow this clean and safe step-by-step process:

---

## ✅ Step-by-Step: Remove Git LFS from a Git Repository

### 🔁 1. **Install Git LFS (temporarily)**

You need Git LFS installed to properly migrate tracked files back into Git:

```bash
git lfs install
```

---

### 🧹 2. **Pull the real content from LFS**

This ensures you have the actual files, not just LFS pointer files:

```bash
git lfs pull
```

---

### 🧬 3. **Migrate all LFS files back into normal Git history**

This rewrites Git history and replaces LFS pointers with actual file contents:

```bash
git lfs migrate export --everything
```

> This will scan all commits and convert LFS files into regular Git objects.

---

### 🚫 4. **Remove LFS tracking**

#### A. Untrack specific files:

```bash
git lfs untrack "path/to/file.ext"
```

#### B. Or untrack all:

Edit `.gitattributes` and delete all lines referencing `filter=lfs`.
Then run:

```bash
git rm --cached .gitattributes
```

And re-add it if needed:

```bash
git add .gitattributes
```

---

### 🚀 5. **Force push updated repo**

Because you've rewritten history, you'll need to force push:

```bash
git add .
git commit -m "Remove LFS and convert files to normal Git"
git push --force
```

> ⚠️ Make sure collaborators know this is a **history rewrite**, and force pushes are involved.

---

### 🧹 6. (Optional) **Delete Git LFS objects from remote**

GitHub (or other LFS providers) may keep the old LFS objects around, even if unused.

To remove them:

* Go to your GitHub repo → Settings → Git LFS → Delete files manually
  OR
* Use GitHub CLI / API to clean them up

---

### 📦 7. **Verify it's clean**

* Check `.gitattributes` is clean (no `filter=lfs`)
* Check LFS is no longer tracking anything:

```bash
git lfs ls-files
```

> Should return nothing.

* Check your files are now regular Git blobs:

```bash
git show HEAD:path/to/your/file
```

---

## ✅ Summary

| Step                                  | Description                   |
| ------------------------------------- | ----------------------------- |
| `git lfs pull`                        | Download real content         |
| `git lfs migrate export --everything` | Convert to normal Git objects |
| Edit `.gitattributes`                 | Remove LFS tracking rules     |
| `git push --force`                    | Rewrite history on remote     |
| Confirm with `git lfs ls-files`       | Should be empty               |

---

Let me know if you’d like a one-liner for cleaning a small project or help safely backing it up before rewriting history.


If `git lfs ls-files` **still lists** `web/public/amplify-ui.css` even **after you've run**:

```bash
git lfs migrate export --everything
```

...then either:

1. The file is still **listed in `.gitattributes`** to be tracked by LFS, **or**
2. Your working tree is clean, but Git history still includes it tracked as LFS.

---

## ✅ Let’s fix it completely — hard reset style

### 🧨 1. Double-check and clean `.gitattributes`

Open `.gitattributes` in the root of your repo. If it still contains:

```
web/public/amplify-ui.css filter=lfs ...
```

Then Git will still *retrack* the file in LFS.

❌ Delete that line (or comment it out).
✅ Then run:

```bash
git rm --cached .gitattributes
git add .gitattributes
```

---

### 🔁 2. Re-run the full LFS removal migration (to ensure all commits are scrubbed)

Run this again to be safe:

```bash
git lfs migrate export --everything --include="web/public/amplify-ui.css"
```

> You can also just migrate the entire repo:

```bash
git lfs migrate export --everything
```

---

### ✅ 3. Untrack the file from LFS explicitly

```bash
git lfs untrack "web/public/amplify-ui.css"
git add .gitattributes
```

You can verify untracking worked:

```bash
git lfs track  # This should NOT list amplify-ui.css
```

---

### 🧼 4. Clean up and re-add the file to Git

```bash
git rm --cached web/public/amplify-ui.css
git add web/public/amplify-ui.css
git commit -m "Fully remove amplify-ui.css from LFS"
```

---

### 🚀 5. Push rewritten history

Because you rewrote history, you need to **force-push**:

```bash
git push --force
```

---

### 🧪 6. Confirm it's 100% clean

```bash
git lfs ls-files
```

✅ It should now return **nothing**

---

## 💡 Optional: Double-check the Git blob

To make sure Git is storing the actual file (not a pointer):

```bash
git show HEAD:web/public/amplify-ui.css | head
```

If it's a **plain CSS file**, you're done.

If it looks like:

```
version https://git-lfs.github.com/spec/v1
oid sha256:...
size 1234
```

→ That’s still a Git LFS pointer file → one of the steps above didn’t stick.

---

Let me know if you'd like to rewrite the whole repo cleanly and start fresh — or keep the full LFS-free version in a new branch.


docker run -d -p 7860:7860 --name starfish-app -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf -v $(pwd)/supervisord.conf:/etc/supervisor/conf.d/supervisord.conf starfish-app

docker build -t starfish-app . 
docker build --no-cache -t your-image-name:your-tag .
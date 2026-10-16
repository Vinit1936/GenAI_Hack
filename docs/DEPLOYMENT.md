# FarmWatch — Production Deployment Guide

This document outlines deployment architecture, infrastructure requirements, containerization procedures, and hosting recommendations for FarmWatch.

---

## 1. Architecture & Hosting Overview

FarmWatch employs a decoupled architecture:
1. **Frontend**: Static single-page application built with React and Vite. Deployed to edge CDNs (Vercel, Netlify, Cloudflare Pages, or AWS S3 + CloudFront).
2. **Backend**: Containerized FastAPI ASGI service with pre-loaded PyTorch CPU weights. Deployed to container hosts (Google Cloud Run, AWS App Runner, Render, Fly.io, or an Ubuntu VPS).

```
  [ Users / Field Agents ]
              |
              +---> [ Edge CDN (Vercel / Cloudflare) ] ---> React UI
              |
              +---> [ Reverse Proxy (Nginx / HTTPS) ]
                            |
                            v
                    [ Docker Container ]
                    FastAPI + PyTorch CPU
                    artifacts/models/farmwatch_resnet18.pth
```

---

## 2. Resource & Hardware Requirements

| Component | Minimum | Recommended Production |
| :--- | :--- | :--- |
| **CPU** | 1 vCPU (x86_64 or ARM64) | 2 vCPUs |
| **RAM** | 1.0 GB | 2.0 GB (accommodates OS + Python + PyTorch + model in memory) |
| **Disk** | 2.0 GB | 10.0 GB (Docker image + layers) |
| **GPU** | Not required (Inference runs in ~25ms on modern CPU) | Optional for massive batch throughput |

---

## 3. Production Environment Variables

### Backend Environment Variables (`.env`)
| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Bind host address for production |
| `PORT` | `8000` | Port on which Uvicorn listens |
| `FARMWATCH_MODEL_PATH` | `artifacts/models/farmwatch_resnet18.pth` | Absolute or relative path to the trained `.pth` checkpoint |
| `FRONTEND_ORIGINS` | `https://farmwatch.example.com` | Strict comma-separated list of allowed CORS origins |
| `MAX_UPLOAD_MB` | `10` | Maximum allowable payload size (in megabytes) |
| `MAX_IMAGE_PIXELS` | `89478485` | Pillow decompression bomb safeguard limit |
| `CONFIDENCE_WARNING_THRESHOLD` | `0.60` | Softmax probability threshold below which a warning banner is triggered |

### Frontend Environment Variables (`frontend/.env.production`)
| Variable | Value | Description |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `https://api.farmwatch.example.com` | Public HTTPS origin of the backend FastAPI service |

---

## 4. Containerization with Docker

A production multi-stage Docker build encapsulates Python 3.11, PyTorch CPU, dependencies, and model weights into a hardened, non-root container.

### Step 1: Build the Backend Docker Image
```bash
docker build -t farmwatch-api:latest .
```

### Step 2: Run Container Locally
```bash
docker run -d \
  -p 8000:8000 \
  -e FRONTEND_ORIGINS="http://localhost:5173,https://farmwatch.example.com" \
  --name farmwatch \
  farmwatch-api:latest
```

### Step 3: Verify Container Health
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{"status":"healthy","model_loaded":true,"model_architecture":"resnet18","device":"cpu","classes_count":5}
```

---

## 5. Docker Compose Deployment

For single-host full-stack deployment (VPS, EC2, or local staging), use `docker-compose.yml`:

```bash
# Start backend and frontend simultaneously
docker-compose up -d --build

# View runtime logs
docker-compose logs -f

# Gracefully tear down
docker-compose down
```

---

## 6. Frontend Production Build & CDN Deployment

### Building Static Assets
```bash
cd frontend
npm ci
npm run build
```

This generates optimized client bundles in `frontend/dist/`:
- `index.html` (~1 kB)
- `assets/index-[hash].css` (~13 kB)
- `assets/index-[hash].js` (~236 kB)

### Deploying to Vercel / Netlify
1. Connect the GitHub repository.
2. Set Root Directory to `frontend`.
3. Build Command: `npm run build`.
4. Output Directory: `dist`.
5. Set Environment Variable: `VITE_API_BASE_URL=https://your-api-domain.com`.

---

## 7. Security & Hardening Checklist

- [x] **Non-Root Execution:** Container runs under a dedicated unprivileged user (`appuser`).
- [x] **CORS Origin Whitelisting:** Explicitly list allowed frontend domains; never use wildcard `*` with credentials.
- [x] **Memory Bounding:** Image payload sizes are strictly validated before full buffer decoding.
- [x] **Decompression Bomb Protection:** Pillow's `MAX_IMAGE_PIXELS` setting prevents memory exhaustion attacks from malicious image formats.
- [x] **HTTPS Enforcement:** Always place the API behind an SSL termination proxy (Cloudflare, Caddy, or Nginx) in production.
- [x] **Stateless Privacy:** Uploaded photos are decoded in volatile memory only and are never saved to persistent storage or databases.

# FastVideo with Ray Serve Backend and Gradio Frontend

This setup provides a scalable web application for FastVideo inference using Ray Serve as the backend and Gradio as the frontend.

## Architecture

- **Backend**: Ray Serve handles video generation requests with GPU acceleration
- **Frontend**: Gradio provides a user-friendly web interface
- **Communication**: HTTP REST API between frontend and backend

## Features

- ✅ Scalable backend with Ray Serve
- ✅ GPU-accelerated video generation
- ✅ User-friendly Gradio interface
- ✅ Health monitoring and error handling
- ✅ All original functionality preserved
- ✅ Easy deployment and management

## Installation

1. Install the additional dependencies:

```bash
pip install -r requirements_ray_serve.txt
```

2. Ensure you have the FastVideo model available (the default is `FastVideo/FastHunyuan-diffusers`)

## Usage

### Option 1: Start Both Services Together (Recommended)

Use the startup script to launch both backend and frontend:

```bash
python start_ray_serve_app.py
```

This will:
- Start the Ray Serve backend on port 8000
- Start the Gradio frontend on port 7860
- Monitor both services and provide unified logging
- Handle graceful shutdown with Ctrl+C

### Option 2: Start Services Separately

#### Start Backend Only

```bash
python ray_serve_backend.py --model_path FastVideo/FastHunyuan-diffusers --output_path outputs
```

#### Start Frontend Only

```bash
python gradio_frontend.py --backend_url http://localhost:8000 --model_path FastVideo/FastHunyuan-diffusers
```

## Configuration

### Command Line Arguments

#### Startup Script (`start_ray_serve_app.py`)

- `--model_path`: Path to the FastVideo model (default: `FastVideo/FastHunyuan-diffusers`)
- `--output_path`: Directory to save generated videos (default: `outputs`)
- `--backend_host`: Backend host to bind to (default: `0.0.0.0`)
- `--backend_port`: Backend port (default: `8000`)
- `--frontend_host`: Frontend host to bind to (default: `0.0.0.0`)
- `--frontend_port`: Frontend port (default: `7860`)
- `--skip_backend_check`: Skip backend health check

#### Backend (`ray_serve_backend.py`)

- `--model_path`: Path to the FastVideo model
- `--output_path`: Directory to save generated videos
- `--host`: Host to bind to
- `--port`: Port to bind to

#### Frontend (`gradio_frontend.py`)

- `--backend_url`: URL of the Ray Serve backend
- `--model_path`: Path to the model (for default parameters)
- `--host`: Host to bind to
- `--port`: Port to bind to

### Environment Variables

You can also set these environment variables:

- `FASTVIDEO_MODEL_PATH`: Path to the FastVideo model
- `FASTVIDEO_OUTPUT_PATH`: Directory to save generated videos
- `RAY_SERVE_HOST`: Backend host
- `RAY_SERVE_PORT`: Backend port
- `GRADIO_HOST`: Frontend host
- `GRADIO_PORT`: Frontend port

## API Endpoints

### Backend API (Ray Serve)

- `GET /health`: Health check endpoint
- `POST /generate_video`: Video generation endpoint

#### Video Generation Request

```json
{
  "prompt": "A beautiful sunset over the ocean",
  "negative_prompt": "blurry, low quality",
  "use_negative_prompt": true,
  "seed": 42,
  "guidance_scale": 7.5,
  "num_frames": 21,
  "height": 512,
  "width": 512,
  "num_inference_steps": 20,
  "randomize_seed": false
}
```

#### Video Generation Response

```json
{
  "output_path": "/path/to/generated/video.mp4",
  "seed": 42,
  "success": true,
  "error_message": null
}
```

## Deployment

### Local Development

1. Start the application:
   ```bash
   python start_ray_serve_app.py
   ```

2. Access the frontend at: `http://localhost:7860`
3. Access the backend API at: `http://localhost:8000`

### Production Deployment

For production deployment, consider:

1. **Load Balancing**: Use a reverse proxy (nginx, traefik) in front of the services
2. **Monitoring**: Add monitoring and logging (Prometheus, Grafana)
3. **Scaling**: Configure Ray Serve for horizontal scaling
4. **Security**: Add authentication and rate limiting
5. **Storage**: Use shared storage for video outputs

### Docker Deployment

Create a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements_ray_serve.txt .
RUN pip install -r requirements_ray_serve.txt

# Copy application files
COPY . .

# Expose ports
EXPOSE 8000 7860

# Start the application
CMD ["python", "start_ray_serve_app.py"]
```

## Troubleshooting

### Common Issues

1. **Backend not starting**: Check GPU availability and Ray installation
2. **Frontend can't connect**: Verify backend URL and network connectivity
3. **Video generation fails**: Check model path and GPU memory
4. **Port conflicts**: Change ports using command line arguments

### Logs

- Backend logs are prefixed with `[BACKEND]`
- Frontend logs are prefixed with `[FRONTEND]`
- Use `--skip_backend_check` if you need to debug startup issues

### Performance Tuning

- Adjust `num_replicas` in the Ray Serve deployment for scaling
- Configure `max_concurrent_queries` based on GPU memory
- Use multiple GPUs by modifying `ray_actor_options`

## Migration from Original Gradio Demo

The new setup maintains full compatibility with the original functionality:

1. All parameters and options are preserved
2. The same example prompts are included
3. The UI layout and behavior are identical
4. Video generation quality is the same

The main differences are:
- Backend processing is now handled by Ray Serve
- Better error handling and status monitoring
- Scalable architecture for production use
- Separation of concerns between frontend and backend

## Contributing

To extend this setup:

1. Add new endpoints to `ray_serve_backend.py`
2. Update the frontend in `gradio_frontend.py`
3. Modify the startup script if needed
4. Update this README with new features 
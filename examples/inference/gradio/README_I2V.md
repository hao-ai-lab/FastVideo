# FastVideo Dual Model Setup (T2V + I2V)

This document describes the dual model functionality that has been added to the FastVideo Gradio app, supporting both Text-to-Video (T2V) and Image-to-Video (I2V) generation modes with specialized models.

## Overview

The Gradio app now supports both Text-to-Video (T2V) and Image-to-Video (I2V) generation modes using specialized models:
- **T2V Model**: `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` for text-to-video generation
- **I2V Model**: `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` for image-to-video generation

Users can switch between modes using the tabbed interface and upload images for I2V generation.

## Model Configuration

### T2V Model
- **Model**: `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
- **Purpose**: Text-to-video generation
- **Default Parameters**: Optimized for text prompts

### I2V Model  
- **Model**: `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
- **Purpose**: Image-to-video generation
- **Default Parameters**: Optimized for image animation

## Changes Made

### Backend Changes (`ray_serve_backend.py`)

1. **Added dual model support**:
   - Separate model paths for T2V and I2V
   - Automatic model selection based on request type
   - Independent model initialization

2. **Updated `VideoGenerationRequest`**:
   - Added `model_type` field ("t2v" or "i2v")
   - Added `image_path` field for I2V input

3. **Enhanced `FastVideoAPI` class**:
   - Dual model initialization (`t2v_generator` and `i2v_generator`)
   - Separate default parameters for each model
   - Automatic model selection in `generate_video` method

### Frontend Changes (`gradio_frontend.py`)

1. **Tabbed interface**:
   - "Text-to-Video" tab for T2V generation
   - "Image-to-Video" tab for I2V generation

2. **Automatic model selection**:
   - T2V tab uses T2V model automatically
   - I2V tab uses I2V model automatically
   - Model type sent in API requests

3. **Separate event handlers**:
   - `handle_t2v_generation` for text-to-video
   - `handle_i2v_generation` for image-to-video

## Usage

### Starting the Application

1. **Using the combined startup script (recommended)**:
   ```bash
   python start_ray_serve_app.py
   ```

2. **Manual startup**:
   ```bash
   # Start backend
   python ray_serve_backend.py \
     --t2v_model_path "FastVideo/FastWan2.1-T2V-1.3B-Diffusers" \
     --i2v_model_path "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
   
   # Start frontend
   python gradio_frontend.py --backend_url "http://localhost:8000"
   ```

### Using T2V Mode

1. Navigate to the "Text-to-Video" tab
2. Enter a text prompt describing the video you want to generate
3. Adjust advanced parameters if needed
4. Click "Run" to generate the video

### Using I2V Mode

1. Navigate to the "Image-to-Video" tab
2. Upload an image using the image upload component
3. Enter a prompt describing how the image should animate
4. Adjust advanced parameters if needed
5. Click "Run" to generate the video

### Example Prompts

**T2V Examples**:
- "A hand enters the frame, pulling a sheet of plastic wrap over three balls of dough placed on a wooden surface."
- "A vintage train snakes through the mountains, its plume of white steam rising dramatically against the jagged peaks."

**I2V Examples**:
- "The image comes to life with subtle movement, the scene gently animating while maintaining the original composition and mood."
- "The static image transforms into a dynamic scene with natural motion, preserving the original lighting and atmosphere."

## Testing

A comprehensive test script is provided to verify both T2V and I2V functionality:

```bash
python test_i2v.py
```

This script:
- Tests backend health
- Tests T2V functionality with text prompts
- Tests I2V functionality with image uploads
- Verifies response formats for both modes
- Cleans up test files

## Technical Details

### Backend API Changes

The `/generate_video` endpoint now accepts:

```json
{
  "prompt": "Animation description",
  "model_type": "t2v",  // or "i2v"
  "image_path": "/path/to/input/image.png",  // for I2V
  // ... other parameters
}
```

### Model Selection Logic

- **T2V Mode**: Uses `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
- **I2V Mode**: Uses `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
- **Automatic Selection**: Based on presence of `image_path` and `model_type`

### Memory Management

- Both models are loaded independently
- Automatic cleanup prevents memory leaks
- Temporary files are cleaned up after processing

## Configuration Options

### Command Line Arguments

**Backend**:
- `--t2v_model_path`: Path to T2V model
- `--i2v_model_path`: Path to I2V model
- `--output_path`: Output directory
- `--host`, `--port`: Server configuration

**Frontend**:
- `--backend_url`: Backend API URL
- `--t2v_model_path`, `--i2v_model_path`: Model paths (for reference)
- `--host`, `--port`: Server configuration

## Troubleshooting

1. **Model Loading Issues**: Ensure both models are accessible
2. **Memory Issues**: The backend includes automatic cleanup
3. **Image Upload Failures**: Check image format and size
4. **Generation Failures**: Check backend logs for detailed errors

## Performance Considerations

- **Model Loading**: Both models are loaded at startup
- **Memory Usage**: Higher memory requirements due to dual models
- **Generation Time**: I2V may take longer due to larger model size
- **GPU Requirements**: Ensure sufficient VRAM for both models

## Future Enhancements

Potential improvements:
- Model switching without restart
- Batch processing for both modes
- Advanced image preprocessing
- Progress indicators
- Result caching and history
- Model-specific parameter optimization 
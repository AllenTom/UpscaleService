import io
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Request, UploadFile, File
from fastapi import HTTPException
from starlette.responses import StreamingResponse
import shared
from upscaler import Upscaler

app = FastAPI()

arg_device = os.getenv('DEVICE', 'cuda')
arg_gpu_id = os.getenv('GPU_ID', None)

if arg_device == 'cuda':
    if torch.cuda.is_available():
        shared.device = torch.device('cuda')
        print('use cuda now')
        if arg_gpu_id is not None:
            shared.gpu_id = int(arg_gpu_id)
            print('use gpu id:', shared.gpu_id)
    else:
        shared.device = torch.device('cpu')
        print('cuda is not available, use cpu now')
else:
    shared.device = torch.device('cpu')
    print('use cpu now')

def make_output(data=None, success=True, error=None):
    return {
        "data": data,
        "error": error,
        "success": success
    }

@app.get("/info")
async def info():
    return make_output({
        "name": "Image Upscaler API",
    })
@app.post("/upscale")
async def upscale(request: Request, file: UploadFile = File(...), out_scale: float = 1.5,
                  model_name="RealESRGAN_x4plus", face_enhance=""):
    # Convert the uploaded file to a numpy array
    image_data = await file.read()
    nparr = np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Check the file format
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Determine the output format based on the input format
    if file.content_type == "image/jpeg":
        output_format = ".jpg"
    elif file.content_type == "image/png":
        output_format = ".png"

    upscaler = Upscaler(
        device=shared.device,
        gpu_id=shared.gpu_id
    )
    face_enhancer = None
    if face_enhance and len(face_enhance) > 0:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=out_scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upscaler)
    # Use the upscaler to upscale the image
    if face_enhancer:
        _, _, upscaled_img = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        upscaled_img = upscaler.upscale(img, outscale=out_scale, model_name=model_name)

    # Convert the upscaled numpy array back to a byte array
    is_success, buffer = cv2.imencode(output_format, upscaled_img)
    io_buf = io.BytesIO(buffer)

    # Convert the BytesIO object to a StreamingResponse
    return StreamingResponse(io_buf, media_type=file.content_type)

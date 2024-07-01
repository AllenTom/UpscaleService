import io

import cv2
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi import HTTPException
from starlette.responses import StreamingResponse

from upscaler import Upscaler

app = FastAPI()

@app.post("/upscale")
async def upscale(request: Request, file: UploadFile = File(...), out_scale: float = 1.5,
                  model_name="RealESRGAN_x4plus",face_enhance = ""):
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

    upscaler = Upscaler()
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
        upscaled_img = upscaler.upscale(img, outscale=out_scale, gpu_id=0, model_name=model_name)

    # Convert the upscaled numpy array back to a byte array
    is_success, buffer = cv2.imencode(output_format, upscaled_img)
    io_buf = io.BytesIO(buffer)

    # Convert the BytesIO object to a StreamingResponse
    return StreamingResponse(io_buf, media_type=file.content_type)

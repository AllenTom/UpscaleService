# UpscaleService
api service for upscale image,base on [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

## Installation

```shell
#install pytorch cuda
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
#install cpu
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
# install other dependencies
pip install -r requirements.txt
```

## Run

use it with uvicorn,more info see [uvicorn](https://www.uvicorn.org/)

```shell
python -m uvicorn main:app --reload 
```

## API

### upscale

url: `/upscale`

request body:

| key  | value | type | description |
|------|-------|------|-------------|
| file | file  | file | image file  |

url params:

| key          | value                | type  | description  |
|--------------|----------------------|-------|--------------|
| out_scale    | float                | float | output scale |
| model_name   | str                  | str   | model name   |
| face_enhance | any string,if enable | str   | face enhance |

model_name:
- RealESRGAN_x4plus
- RealESRNet_x4plus
- RealESRGAN_x4plus_anime_6B
- RealESRGAN_x2plus
- realesr-animevideov3
- realesr-general-x4v3

response:

binary image file


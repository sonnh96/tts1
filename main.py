from fastapi import FastAPI
from pydantic import BaseModel
from XTTS import XTTS
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from fastapi.responses import FileResponse

app = FastAPI()

# Define a folder for static files
STATIC_DIR = Path("output")
STATIC_DIR.mkdir(exist_ok=True)  # Ensure the folder exists
REFERENCE_AUDIO = "2_20231227_1742037149.wav"


XTTS_MODEL = XTTS()

class Data(BaseModel):
    content: str
    
    
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/tts")
async def gen_audio(item: Data):
    input_text = item.content

    output_file = XTTS_MODEL.generate_speech(
        text=input_text,
        speaker_audio_file=REFERENCE_AUDIO,
        language="Tiếng Việt",
        normalize_text=True,
        verbose=True,
        output_chunks=False,  # Disable individual chunk saving for speed
        batch_size=6,  # Process more chunks in parallel
        enable_caching=True  # Enable caching for repeated requests
    )
    file_url = f"/static/{os.path.basename(output_file)}"
    return FileResponse(output_file, media_type="audio/wav", filename=os.path.basename(output_file))


@app.get("/cache-info")
async def get_cache_info():
    """Get information about current cache status."""
    return XTTS_MODEL.get_cache_info()


@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches to free memory."""
    XTTS_MODEL.clear_cache()
    return {"message": "Cache cleared successfully"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8009, reload=False)
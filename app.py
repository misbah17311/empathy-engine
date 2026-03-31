"""
The Empathy Engine — FastAPI Web Application
Provides a web UI and API endpoint for emotion-based text-to-speech.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from emotion_detector import detect_emotion, EMOTION_CATEGORIES
from voice_synthesizer import synthesize_speech, get_voice_parameters, EMOTION_VOICE_MAP

app = FastAPI(title="The Empathy Engine", description="Giving AI a Human Voice")

# Setup templates and output directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Serve generated audio files
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main web interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "emotion_categories": EMOTION_CATEGORIES,
    })


@app.post("/synthesize", response_class=HTMLResponse)
async def synthesize(request: Request, text: str = Form(...)):
    """Accept text input, detect emotion, generate modulated speech."""
    if not text.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "emotion_categories": EMOTION_CATEGORIES,
            "error": "Please enter some text.",
        })

    # Step 1: Detect emotion
    emotion = detect_emotion(text)

    # Step 2: Synthesize speech with emotion-modulated parameters
    result = synthesize_speech(
        text=text,
        emotion_category=emotion["category"],
        intensity=emotion["intensity"],
        output_dir=OUTPUT_DIR,
    )

    # Step 3: Get parameter comparison (neutral vs modulated)
    neutral_params = get_voice_parameters("neutral", 0.0)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "emotion_categories": EMOTION_CATEGORIES,
        "text": text,
        "emotion": emotion,
        "result": result,
        "neutral_params": neutral_params,
        "audio_url": f"/output/{os.path.basename(result['file_path'])}",
        "emotion_map": EMOTION_VOICE_MAP,
    })


@app.post("/api/synthesize")
async def api_synthesize(text: str = Form(...)):
    """
    API endpoint for programmatic access.
    Returns JSON with emotion analysis and audio file URL.
    """
    if not text.strip():
        return JSONResponse({"error": "Text input is required."}, status_code=400)

    emotion = detect_emotion(text)
    result = synthesize_speech(
        text=text,
        emotion_category=emotion["category"],
        intensity=emotion["intensity"],
        output_dir=OUTPUT_DIR,
    )

    return JSONResponse({
        "text": text,
        "emotion": emotion,
        "voice_parameters": result["parameters"],
        "ssml": result["ssml"],
        "audio_url": f"/output/{os.path.basename(result['file_path'])}",
    })


@app.get("/api/emotions")
async def api_emotions():
    """Return the emotion-to-voice mapping configuration."""
    return JSONResponse({
        "categories": EMOTION_CATEGORIES,
        "voice_mapping": EMOTION_VOICE_MAP,
    })

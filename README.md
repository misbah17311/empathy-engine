# The Empathy Engine — Challenge 1 🎙️

**Giving AI a Human Voice — Emotion-Driven Text-to-Speech**

> **Live Demo:** https://empathy-engine-hbj6.onrender.com

A service that analyzes the emotional content of text and generates speech audio with dynamically modulated vocal parameters (voice selection, speed, pitch, volume) to match the detected emotion. Uses OpenAI's TTS API for natural, high-quality voices with ffmpeg post-processing for pitch and volume modulation. Moving beyond monotonic TTS delivery to achieve emotional resonance.

---

## Features

- **Emotion Detection**: Classifies text into 7 granular emotion categories using VADER sentiment analysis
- **Vocal Parameter Modulation**: Adjusts 4 voice parameters (voice, speed, pitch shift, volume) based on detected emotion
- **High-Quality TTS**: Uses OpenAI's Text-to-Speech API for natural human-like voice output
- **Intensity Scaling**: Emotion strength proportionally scales voice parameter changes
- **SSML Integration**: Generates Speech Synthesis Markup Language with prosody, emphasis, and break tags for fine-grained voice control
- **Web Interface**: Clean, modern UI with embedded audio player, parameter visualization, and SSML preview
- **REST API**: Programmatic access via JSON API endpoints
- **CLI**: Command-line interface with interactive mode

## Supported Emotions

| Emotion | Voice | Speed | Pitch | Volume | Description |
|---------|-------|-------|-------|--------|-------------|
| 😊 Happy | Nova | ↑ Faster | ↑ Higher | ↑ Louder | Positive, upbeat text |
| 😢 Sad | Onyx | ↓ Slower | ↓ Lower | ↓ Quieter | Negative, melancholic text |
| 😠 Angry | Onyx | ↑ Faster | ↓ Lower | ↑ Louder | Frustrated, aggressive text |
| 😲 Surprised | Nova | ↑ Faster | ↑↑ Higher | ↑ Louder | Excited, astonished text |
| 🤔 Inquisitive | Shimmer | ↓ Slightly slower | ↑ Rising | — Normal | Questions, curiosity |
| 😟 Concerned | Fable | ↓ Slower | ↓ Slightly lower | ↓ Slightly quieter | Worried, anxious text |
| 😐 Neutral | Alloy | — Baseline | — Baseline | — Baseline | Factual, informational text |

---

## Setup & Installation

### Prerequisites

- **Python 3.8+**
- **ffmpeg** (for audio post-processing)
- **OpenAI API key** ([get one here](https://platform.openai.com/api-keys))

### Step-by-Step Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd empathy-engine
   ```

2. **Install ffmpeg** (Linux):
   ```bash
   sudo apt-get install -y ffmpeg
   ```
   > On macOS: `brew install ffmpeg`. On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html).

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application**:

   **Web Interface (recommended):**
   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   Then open [http://localhost:8000](http://localhost:8000) in your browser.

   **CLI — Single text:**
   ```bash
   python cli.py "I just got promoted! This is the best day ever!"
   ```

   **CLI — Custom output file:**
   ```bash
   python cli.py "I'm worried about the deadline." -o worried.mp3
   ```

   **CLI — Interactive mode:**
   ```bash
   python cli.py --interactive
   ```

---

## API Usage

### POST `/api/synthesize`

Analyze text and generate emotion-modulated speech.

**Request:**
```bash
curl -X POST http://localhost:8000/api/synthesize \
  -d "text=I just got promoted! This is the best day ever!"
```

**Response:**
```json
{
  "text": "I just got promoted! This is the best day ever!",
  "emotion": {
    "category": "surprised",
    "intensity": 0.82,
    "compound": 0.8217,
    "scores": {"neg": 0.0, "neu": 0.569, "pos": 0.431, "compound": 0.8217}
  },
  "voice_parameters": {"voice": "nova", "speed": 1.12, "pitch_shift": 2.0, "volume": 1.05},
    "audio_url": "/output/speech_surprised_abc12345.mp3"
}
```

### GET `/api/emotions`

Returns the emotion-to-voice parameter mapping configuration.

---

## Design Choices

### Emotion Detection: Why VADER?

VADER (Valence Aware Dictionary and sEntiment Reasoner) was chosen because:
- **No API keys or internet required** — fully offline
- **Optimized for social media and conversational text** — handles slang, emoticons, punctuation emphasis
- **Provides a compound score** (-1 to +1) which enables intensity scaling
- **Fast** — no model loading time, instant classification

The compound score is mapped to 7 granular categories using both the score ranges and text heuristics (e.g., question marks for inquisitive, specific keywords for concerned/angry).

### Emotion-to-Voice Mapping Logic

Each emotion maps to a specific **OpenAI voice** (for tonal character) and defines **delta values** for speed, pitch shift, and volume:

```
Baseline: Speed=1.0x, Pitch Shift=0 semitones, Volume=1.0

Happy:       Voice=Nova,    Speed +0.10,  Pitch +1.5st,  Volume +0.05
Sad:         Voice=Onyx,    Speed -0.15,  Pitch -1.5st,  Volume -0.10
Angry:       Voice=Onyx,    Speed +0.08,  Pitch -1.0st,  Volume +0.10
Surprised:   Voice=Nova,    Speed +0.12,  Pitch +2.0st,  Volume +0.05
Inquisitive: Voice=Shimmer, Speed -0.05,  Pitch +1.0st,  Volume  0.00
Concerned:   Voice=Fable,   Speed -0.10,  Pitch -0.5st,  Volume -0.05
```

**Intensity Scaling**: The deltas are multiplied by the intensity score (0.0–1.0). For example, "This is good" (intensity: 0.44) gets a smaller speed increase than "This is the best news ever!" (intensity: 0.86). This creates natural variation rather than binary emotion switching.

### TTS Engine: Why OpenAI TTS + ffmpeg?

**OpenAI TTS** was chosen as the primary speech engine because:
- **Natural, human-like voices** — far superior quality compared to traditional TTS engines
- **Multiple voice personas** (alloy, echo, fable, onyx, nova, shimmer) — allows mapping emotions to distinct vocal characters
- **Native speed control** (0.25x–4.0x) — built-in rate modulation

**ffmpeg** handles post-processing for parameters OpenAI doesn't expose:
- **Pitch shifting** via `asetrate` filter — shifts audio by semitones while maintaining duration with `atempo` compensation
- **Volume adjustment** via `volume` filter — scales amplitude for emotional intensity

This hybrid approach gives us **4 distinct vocal parameters** (voice, speed, pitch, volume) with studio-quality base audio.

### SSML Integration

The system generates Speech Synthesis Markup Language (SSML) to add fine-grained control beyond the base vocal parameters:

- **`<prosody>`** — Wraps each sentence with emotion-appropriate rate and pitch attributes (e.g., `rate="fast" pitch="low"` for angry)
- **`<emphasis>`** — Applies emphasis to emotion-relevant keywords. Each emotion has a curated word list (e.g., angry emphasizes "frustrated", "terrible", "worst") with an appropriate level (strong/moderate/reduced)
- **`<break>`** — Inserts pauses between sentences calibrated to the emotion (angry: 250ms short breaks, sad: 600ms long pauses, neutral: 350ms)

Example output for angry text:
```xml
<prosody rate="fast" pitch="low">
  I am so <emphasis level="strong">frustrated</emphasis> with this
  <emphasis level="strong">terrible</emphasis> service.
</prosody>
<break time="250ms"/>
```

espeak-compatible SSML is generated and displayed in both the web UI and CLI as a bonus demonstration of emotion-to-markup mapping. The SSML showcases how vocal parameters would be expressed in standard speech synthesis markup.

---

## Project Structure

```
empathy-engine/
├── app.py                  # FastAPI web application
├── cli.py                  # Command-line interface
├── emotion_detector.py     # VADER-based emotion detection
├── voice_synthesizer.py    # OpenAI TTS + ffmpeg vocal modulation
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── templates/
│   └── index.html          # Web UI template
└── output/                 # Generated audio files (auto-created)
```

## Tech Stack

- **Python 3.8+**
- **VADER Sentiment** — Emotion/sentiment analysis
- **OpenAI TTS API** — High-quality text-to-speech with voice selection and speed control
- **ffmpeg** — Audio post-processing for pitch shifting and volume adjustment
- **FastAPI** — Web framework and REST API
- **Jinja2** — HTML templating
- **Uvicorn** — ASGI server

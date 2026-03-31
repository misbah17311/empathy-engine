"""
Voice Synthesizer Module
Uses OpenAI TTS for high-quality natural speech generation.
Post-processes audio with ffmpeg to modulate pitch and volume.
Retains SSML generation for demonstrable emotion-to-voice mapping.
"""

import os
import re
import uuid
import subprocess
from openai import OpenAI

client = None


def _get_client() -> OpenAI:
    """Lazy-initialize the OpenAI client."""
    global client
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it with: export OPENAI_API_KEY=your_key_here"
            )
        client = OpenAI(api_key=api_key)
    return client


# Default voice parameters (baseline for neutral speech)
DEFAULT_SPEED = 1.0       # OpenAI TTS speed: 0.25-4.0, 1.0 = normal
DEFAULT_VOLUME = 1.0      # Post-process volume multiplier
DEFAULT_PITCH_SHIFT = 0   # Semitones shift via ffmpeg (0 = no change)

# Emotion-to-voice mapping
# voice: OpenAI voice selection (different tones/timbres)
# speed_delta: adjustment to TTS speed
# pitch_shift: semitones to shift via ffmpeg (-12 to +12)
# volume_delta: volume multiplier adjustment
EMOTION_VOICE_MAP = {
    "happy": {
        "voice": "nova",
        "speed_delta": +0.10,
        "pitch_shift": +1.5,
        "volume_delta": +0.05,
    },
    "sad": {
        "voice": "onyx",
        "speed_delta": -0.15,
        "pitch_shift": -1.5,
        "volume_delta": -0.10,
    },
    "angry": {
        "voice": "onyx",
        "speed_delta": +0.08,
        "pitch_shift": -1.0,
        "volume_delta": +0.10,
    },
    "surprised": {
        "voice": "nova",
        "speed_delta": +0.12,
        "pitch_shift": +2.0,
        "volume_delta": +0.05,
    },
    "inquisitive": {
        "voice": "shimmer",
        "speed_delta": -0.05,
        "pitch_shift": +1.0,
        "volume_delta": 0.0,
    },
    "concerned": {
        "voice": "fable",
        "speed_delta": -0.10,
        "pitch_shift": -0.5,
        "volume_delta": -0.05,
    },
    "neutral": {
        "voice": "alloy",
        "speed_delta": 0,
        "pitch_shift": 0,
        "volume_delta": 0.0,
    },
}


def get_voice_parameters(emotion_category: str, intensity: float) -> dict:
    """
    Calculate voice parameters based on emotion and intensity.

    Returns:
        dict with keys: voice, speed, pitch_shift, volume
    """
    mapping = EMOTION_VOICE_MAP.get(emotion_category, EMOTION_VOICE_MAP["neutral"])

    speed = round(DEFAULT_SPEED + mapping["speed_delta"] * intensity, 2)
    pitch_shift = round(DEFAULT_PITCH_SHIFT + mapping["pitch_shift"] * intensity, 1)
    volume = round(DEFAULT_VOLUME + mapping["volume_delta"] * intensity, 2)

    speed = max(0.25, min(4.0, speed))
    pitch_shift = max(-12, min(12, pitch_shift))
    volume = max(0.1, min(2.0, volume))

    return {
        "voice": mapping["voice"],
        "speed": speed,
        "pitch_shift": pitch_shift,
        "volume": volume,
    }


def synthesize_speech(text: str, emotion_category: str, intensity: float, output_dir: str = "output") -> dict:
    """
    Generate a speech audio file with emotion-modulated vocal parameters.
    Uses OpenAI TTS for natural voice, then ffmpeg for pitch/volume adjustment.
    """
    os.makedirs(output_dir, exist_ok=True)

    params = get_voice_parameters(emotion_category, intensity)
    ssml_text = generate_ssml(text, emotion_category, intensity)

    uid = uuid.uuid4().hex[:8]
    raw_file = os.path.join(output_dir, f"raw_{uid}.mp3")
    final_file = os.path.join(output_dir, f"speech_{emotion_category}_{uid}.mp3")

    # Step 1: Generate speech with OpenAI TTS
    _generate_with_openai(text, params["voice"], params["speed"], raw_file)

    # Step 2: Post-process with ffmpeg for pitch and volume
    _postprocess_audio(raw_file, final_file, params["pitch_shift"], params["volume"])

    # Clean up raw file
    if os.path.exists(raw_file):
        os.remove(raw_file)

    return {
        "file_path": final_file,
        "parameters": params,
        "ssml": ssml_text,
    }


def _generate_with_openai(text: str, voice: str, speed: float, output_path: str):
    """Generate speech using OpenAI TTS API."""
    response = _get_client().audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        speed=speed,
        response_format="mp3",
    )
    response.stream_to_file(output_path)


def _get_sample_rate(input_path: str) -> int:
    """Detect the sample rate of an audio file using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=sample_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        capture_output=True, text=True
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 24000  # OpenAI TTS default


def _postprocess_audio(input_path: str, output_path: str, pitch_shift: float, volume: float):
    """
    Adjust pitch and volume using ffmpeg.
    Uses rubberband filter for clean pitch shifting without speed change.
    """
    filter_parts = []

    if abs(pitch_shift) > 0.1:
        # Use asetrate to shift pitch, then aresample to restore original rate
        sr = _get_sample_rate(input_path)
        pitch_ratio = 2 ** (pitch_shift / 12.0)
        filter_parts.append(f"asetrate={sr}*{pitch_ratio:.6f}")
        filter_parts.append(f"aresample={sr}")
        # atempo to compensate speed change from pitch shift
        tempo = 1.0 / pitch_ratio
        while tempo < 0.5:
            filter_parts.append("atempo=0.5")
            tempo /= 0.5
        while tempo > 2.0:
            filter_parts.append("atempo=2.0")
            tempo /= 2.0
        filter_parts.append(f"atempo={tempo:.6f}")

    if abs(volume - 1.0) > 0.01:
        filter_parts.append(f"volume={volume:.2f}")

    cmd = ["ffmpeg", "-y", "-i", input_path]

    if filter_parts:
        cmd.extend(["-af", ",".join(filter_parts)])

    cmd.extend(["-q:a", "2", output_path])

    subprocess.run(cmd, check=True, capture_output=True)


# --- SSML Generation (for display & bonus points) ---

EMOTION_SSML_CONFIG = {
    "happy": {
        "prosody_rate": "fast",
        "prosody_pitch": "high",
        "emphasis_level": "strong",
        "pause_after_sentence": "300ms",
    },
    "sad": {
        "prosody_rate": "slow",
        "prosody_pitch": "low",
        "emphasis_level": "reduced",
        "pause_after_sentence": "600ms",
    },
    "angry": {
        "prosody_rate": "fast",
        "prosody_pitch": "low",
        "emphasis_level": "strong",
        "pause_after_sentence": "250ms",
    },
    "surprised": {
        "prosody_rate": "fast",
        "prosody_pitch": "x-high",
        "emphasis_level": "strong",
        "pause_after_sentence": "400ms",
    },
    "inquisitive": {
        "prosody_rate": "medium",
        "prosody_pitch": "high",
        "emphasis_level": "moderate",
        "pause_after_sentence": "400ms",
    },
    "concerned": {
        "prosody_rate": "slow",
        "prosody_pitch": "low",
        "emphasis_level": "moderate",
        "pause_after_sentence": "500ms",
    },
    "neutral": {
        "prosody_rate": "medium",
        "prosody_pitch": "medium",
        "emphasis_level": "moderate",
        "pause_after_sentence": "350ms",
    },
}

EMPHASIS_WORDS = {
    "happy": ["great", "amazing", "wonderful", "love", "best", "awesome", "fantastic", "excellent", "perfect", "beautiful", "incredible", "happy", "joy", "excited", "promoted"],
    "sad": ["sorry", "unfortunately", "sad", "miss", "lost", "gone", "never", "alone", "terrible", "worst", "disappointed", "failed", "regret"],
    "angry": ["hate", "angry", "furious", "annoyed", "frustrated", "terrible", "worst", "ridiculous", "unacceptable", "stupid", "horrible", "disgusting"],
    "surprised": ["wow", "amazing", "incredible", "unbelievable", "shocking", "unexpected", "suddenly", "really", "actually", "absolutely"],
    "inquisitive": ["why", "how", "what", "when", "where", "think", "wonder", "curious", "perhaps", "maybe"],
    "concerned": ["worry", "worried", "concern", "afraid", "anxious", "nervous", "hopefully", "careful", "risk", "danger", "unsure"],
    "neutral": [],
}


def generate_ssml(text: str, emotion_category: str, intensity: float) -> str:
    """Generate SSML markup for display purposes."""
    config = EMOTION_SSML_CONFIG.get(emotion_category, EMOTION_SSML_CONFIG["neutral"])
    emphasis_words = set(EMPHASIS_WORDS.get(emotion_category, []))

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences or (len(sentences) == 1 and not sentences[0]):
        sentences = [text]

    ssml_parts = []
    for sentence in sentences:
        if not sentence.strip():
            continue

        words = sentence.split()
        processed_words = []
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
            if clean_word in emphasis_words:
                processed_words.append(
                    f'<emphasis level="{config["emphasis_level"]}">{word}</emphasis>'
                )
            else:
                processed_words.append(word)

        processed_sentence = " ".join(processed_words)
        ssml_parts.append(
            f'<prosody rate="{config["prosody_rate"]}" pitch="{config["prosody_pitch"]}">'
            f'{processed_sentence}'
            f'</prosody>'
            f'<break time="{config["pause_after_sentence"]}"/>'
        )

    return "\n".join(ssml_parts)

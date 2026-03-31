"""
The Empathy Engine — CLI Interface
Command-line tool for emotion-based text-to-speech.
"""

import argparse
import sys
import os
from dotenv import load_dotenv
load_dotenv()

from emotion_detector import detect_emotion, EMOTION_CATEGORIES
from voice_synthesizer import synthesize_speech, get_voice_parameters


def main():
    parser = argparse.ArgumentParser(
        description="The Empathy Engine: Emotion-driven Text-to-Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "I just got promoted! This is the best day ever!"
  python cli.py "I'm so frustrated with this terrible service." -o my_audio.mp3
  python cli.py --interactive
        """,
    )
    parser.add_argument("text", nargs="?", help="Text to analyze and speak")
    parser.add_argument("-o", "--output", help="Output file path (default: output/speech_<emotion>_<id>.mp3)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode (loop)")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.text:
        process_text(args.text, args.output)
    else:
        # Prompt for text input
        print("The Empathy Engine — Emotion-Driven Text-to-Speech")
        print("=" * 52)
        text = input("\nEnter text: ").strip()
        if text:
            process_text(text, args.output)
        else:
            print("No text provided.")
            sys.exit(1)


def process_text(text: str, output_path: str = None):
    """Analyze text emotion and generate modulated speech."""
    print(f"\nInput: \"{text}\"")
    print("-" * 50)

    # Detect emotion
    emotion = detect_emotion(text)
    print(f"\n  Emotion:   {emotion['category'].upper()}")
    print(f"  Intensity: {emotion['intensity']} ({int(emotion['intensity'] * 100)}%)")
    print(f"  Compound:  {emotion['compound']}")
    print(f"  Scores:    pos={emotion['scores']['pos']}, neg={emotion['scores']['neg']}, neu={emotion['scores']['neu']}")

    # Get voice parameters
    params = get_voice_parameters(emotion["category"], emotion["intensity"])
    neutral = get_voice_parameters("neutral", 0.0)

    print(f"\n  Voice Parameters:")
    print(f"    Voice:       {params['voice']}")
    print(f"    Speed:       {params['speed']}x (neutral: {neutral['speed']}x, delta: {params['speed'] - neutral['speed']:+.2f})")
    print(f"    Pitch Shift: {params['pitch_shift']} semitones (neutral: {neutral['pitch_shift']}, delta: {params['pitch_shift'] - neutral['pitch_shift']:+.1f})")
    print(f"    Volume:      {params['volume']} (neutral: {neutral['volume']}, delta: {params['volume'] - neutral['volume']:+.2f})")

    # Synthesize speech
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    result = synthesize_speech(
        text=text,
        emotion_category=emotion["category"],
        intensity=emotion["intensity"],
        output_dir=output_dir,
    )

    # Rename if custom output path specified
    if output_path:
        os.rename(result["file_path"], output_path)
        result["file_path"] = output_path

    print(f"\n  SSML Markup:")
    for line in result["ssml"].split("\n"):
        print(f"    {line}")

    print(f"\n  Audio saved to: {result['file_path']}")
    print()


def interactive_mode():
    """Run in a loop, continuously accepting text input."""
    print("The Empathy Engine — Interactive Mode")
    print("=" * 40)
    print("Type text and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            text = input(">> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if text:
                process_text(text)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()

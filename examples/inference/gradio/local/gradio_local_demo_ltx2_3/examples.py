import json
import os

def load_example_prompts():
    examples: list[str] = []
    example_labels: list[str] = []
    prompts_path = os.path.join(
        os.path.dirname(__file__),
        "selected_ltx2_prompts.jsonl",
    )

    try:
        with open(prompts_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prompt = entry.get("video_prompt", "")
                if not isinstance(prompt, str):
                    continue
                prompt = prompt.strip()
                if not prompt:
                    continue
                examples.append(prompt)
                example_labels.append(
                    prompt[:100] + "..." if len(prompt) > 100 else prompt
                )
    except Exception as e:
        print(f"Warning: Could not read {prompts_path}: {e}")

    if not examples:
        # Backward-compatible fallback to validation captions.
        validation_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "distill",
            "LTX2",
            "validation.json",
        )
        try:
            with open(validation_path, encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("data", []):
                caption = entry.get("caption", "")
                if not isinstance(caption, str):
                    continue
                caption = caption.strip()
                if not caption:
                    continue
                examples.append(caption)
                example_labels.append(
                    caption[:100] + "..." if len(caption) > 100 else caption
                )
        except Exception as e:
            print(f"Warning: Could not read {validation_path}: {e}")

    if not examples:
        examples = [
            "A crowded rooftop bar buzzes with energy, the city skyline twinkling like a field of stars in the background."
        ]
        example_labels = ["Crowded rooftop bar at night"]

    return examples, example_labels

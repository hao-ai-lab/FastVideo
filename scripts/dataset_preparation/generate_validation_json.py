import argparse
import json
import os
import random


def generate_merged_validation_json(input_dir, output_file):
    # read in video2caption.json
    with open(os.path.join(input_dir, "videos2caption.json"), "r") as f:
        video2caption = json.load(f)

    # count how many elements are in the list
    num_elements = len(video2caption)
    print(f"Number of elements in video2caption.json: {num_elements}")

    # randomly sample 32 elements from the list
    sampled_elements = random.sample(video2caption, 32)

    # Transform sampled elements into validation.json format
    validation_data = []
    for element in sampled_elements:
        assert element.get("cap") is not None, f"Caption is None for element: {element}"
        caption = element["cap"]
        if isinstance(caption, list):
            caption = caption[0]
            
        validation_entry = {
            "caption": caption,
            "video_path": element.get("path", ""),
            "num_inference_steps": 8,
            "height": 1088,
            "width": 1920,
            "num_frames": 121
        }
        validation_data.append(validation_entry)

    # Create the final validation structure
    validation_json = {
        "data": validation_data
    }

    # Write the validation JSON to the output file
    with open(output_file, "w") as f:
        json.dump(validation_json, f, indent=2)
    
    print(f"Generated validation JSON with {len(validation_data)} entries and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    # dataset_type: "mixkit"
    parser.add_argument("--dataset_type", choices=["merged"], required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_elements", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=77)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    args = parser.parse_args()

    if args.dataset_type == "merged":
        generate_merged_validation_json(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
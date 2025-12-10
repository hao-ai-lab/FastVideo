import argparse
import json
import os
import random


def generate_merged_validation_json(args):
    input_file = args.input_file
    output_validation_file = args.output_validation_file
    
    if args.output_train_file:
        output_train_file = args.output_train_file
    else:
        base, ext = os.path.splitext(input_file)
        output_train_file = f"{base}_train{ext}"

    # read in input json
    print(f"Reading from {input_file}")
    with open(input_file, "r") as f:
        video2caption = json.load(f)

    # count how many elements are in the list
    num_elements = len(video2caption)
    print(f"Number of elements in input file: {num_elements}")

    # randomly sample elements from the list
    num_sample = min(args.num_elements, num_elements)
    indices = set(random.sample(range(num_elements), num_sample))
    
    sampled_elements = []
    remaining_elements = []
    
    for i in range(num_elements):
        if i in indices:
            sampled_elements.append(video2caption[i])
        else:
            remaining_elements.append(video2caption[i])

    # Transform sampled elements into validation.json format
    validation_data = []
    for element in sampled_elements:
        assert element.get("cap") is not None, f"Caption is None for element: {element}"
        validation_entry = {
            "caption": element["cap"],
            "video_path": element.get("path", ""),
            "num_inference_steps": args.num_inference_steps,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames
        }
        validation_data.append(validation_entry)

    # Create the final validation structure
    validation_json = {
        "data": validation_data
    }

    # Write the validation JSON to the output file
    with open(output_validation_file, "w") as f:
        json.dump(validation_json, f, indent=2)
    
    print(f"Generated validation JSON with {len(validation_data)} entries and saved to {output_validation_file}")

    # Write the remaining JSON to the output train file
    with open(output_train_file, "w") as f:
        json.dump(remaining_elements, f, indent=2)
    
    print(f"Saved remaining {len(remaining_elements)} entries to {output_train_file}")


def main():
    parser = argparse.ArgumentParser()
    # dataset_type: "merged"
    parser.add_argument("--dataset_type", choices=["merged"], required=True)
    parser.add_argument("--input_file", type=str, required=True, help="Path to input json file")
    parser.add_argument("--output_validation_file", type=str, required=True, help="Path to output validation json file")
    parser.add_argument("--output_train_file", type=str, help="Path to output train json file (remaining data). Defaults to {input_filename}_train.json")
    parser.add_argument("--num_elements", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=77)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    args = parser.parse_args()

    if args.dataset_type == "merged":
        generate_merged_validation_json(args)


if __name__ == "__main__":
    main()

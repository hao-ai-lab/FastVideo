
import json
import re
import argparse
from pathlib import Path

def extract_scene_info(filename):
    """
    Extract scene information from filename
    Returns tuple of (is_cropped, first_number, second_number)
    """
    is_cropped = filename.startswith('cropped')
    match = re.search(r'scene-(\d+)-(\d+)', filename)
    if match:
        return (is_cropped, int(match.group(1)), int(match.group(2)))
    return (is_cropped, 0, 0)

def rank_scenes(json_data, output_file=None, verbose=False):
    """
    Rank scenes with cropped videos first, then by scene numbers
    """
    scenes = list(json_data['captions'].items())
    sorted_scenes = sorted(scenes, 
                         key=lambda x: (not extract_scene_info(x[0])[0], 
                                      extract_scene_info(x[0])[1], 
                                      extract_scene_info(x[0])[2]))
    
    print("\nScene Rankings:")
    print("-" * 50)
    for i, (filename, data) in enumerate(sorted_scenes, 1):
        is_cropped, num1, num2 = extract_scene_info(filename)
        print(f"\n{i}. {filename}")
        if verbose:
            print(f"   Cropped: {is_cropped}")
            print(f"   Scene numbers: {num1}-{num2}")
            print(f"   Caption length: {len(data['caption'])}")
    
    if output_file:
        sorted_data = {
            "generation_info": json_data["generation_info"],
            "captions": {k: v for k, v in sorted_scenes}
        }
        with open(output_file, 'w') as f:
            json.dump(sorted_data, f, indent=2)
        print(f"\nSorted data saved to: {output_file}")
    
    return sorted_scenes

def main():
    parser = argparse.ArgumentParser(description='Rank video scenes with cropped scenes first, then by scene numbers')
    parser.add_argument('input_file', type=str, help='Input JSON file containing scene captions')
    parser.add_argument('--output', '-o', type=str, help='Output file for sorted JSON (optional)', default=None)
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed information for each scene')

    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: '{args.input_file}' is not a valid JSON file")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    rank_scenes(data, args.output, args.verbose)

if __name__ == "__main__":
    main()
import os
import shutil
import argparse

def process_videos(source_folder, destination_folder="HSH_500", should_move=False):
    """
    Count and optionally move videos where scene number A is less than 1500
    from filenames matching pattern hsh_scene-A-B.mp4
    
    Args:
        source_folder (str): Path to the source folder containing videos
        destination_folder (str): Name of the destination folder
        should_move (bool): Whether to move matching files
        
    Returns:
        tuple: (count of matching videos, list of processed files)
    """
    if should_move:
        dest_path = os.path.join(os.path.dirname(source_folder), destination_folder)
        os.makedirs(dest_path, exist_ok=True)
    
    count = 0
    processed_files = []
    
    for filename in os.listdir(source_folder):
        if not filename.endswith('.mp4'):
            continue
            
        try:
            # Split by underscore and get the scene part
            parts = filename.split('_')
            if len(parts) != 2:
                continue
                
            # Get the scene numbers part (scene-A-B.mp4)
            scene_part = parts[1]
            
            # Split by '-' to get individual numbers
            scene_numbers = scene_part.split('-')
            if len(scene_numbers) != 3:  # scene, A, B.mp4
                continue
                
            # Get A value and remove non-numeric characters
            a_value = scene_numbers[1]
            
            # Convert to integer for comparison
            a_int = int(a_value)
            
            if a_int < 1600:
                count += 1
                processed_files.append(filename)
                
                if should_move:
                    source_file = os.path.join(source_folder, filename)
                    dest_file = os.path.join(dest_path, filename)
                    try:
                        shutil.move(source_file, dest_file)
                        print(f"Moved: {filename}")
                    except Exception as e:
                        print(f"Error moving {filename}: {str(e)}")
                        
        except (ValueError, IndexError) as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return count, processed_files

def main():
    parser = argparse.ArgumentParser(description='Process video files with scene numbers less than 1500')
    parser.add_argument('--source_folder', help='Path to the folder containing video files')
    parser.add_argument('--move', action='store_true', help='Move matching files to destination folder')
    parser.add_argument('--dest_folder', default='HSH_500', help='Destination folder name (default: HSH_500)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.source_folder):
        print(f"Error: Source folder '{args.source_folder}' does not exist")
        return
    
    count, files = process_videos(
        source_folder=args.source_folder,
        destination_folder=args.dest_folder,
        should_move=args.move
    )
    
    print(f"\nFound {count} videos with scene number less than 1500:")

    
    if args.move:
        print(f"\nFiles were moved to: {os.path.join(os.path.dirname(args.source_folder), args.dest_folder)}")
    
if __name__ == "__main__":
    main()
import os
import glob
import re

def extract_summary(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Look for the summary block
    match = re.search(r'={10,}\s*AVERAGE TIMING SUMMARY\s*={10,}(.*?)={10,}', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def main():
    log_dir = '/home/d1su/codes/FastVideo-demo/profile-ltx2/profile_results_02-15'
    log_files = glob.glob(os.path.join(log_dir, '*.log'))
    
    print(f"Found {len(log_files)} log files.")

    for log_file in log_files:
        summary = extract_summary(log_file)
        if summary:
            # Create a summary filename
            base_name = os.path.basename(log_file)
            summary_filename = os.path.join(log_dir, f"{base_name}.summary.md")
            
            with open(summary_filename, 'w') as f:
                f.write(f"# Latency Summary for {base_name}\n\n")
                f.write("```csv\n")
                f.write(summary)
                f.write("\n```\n")
                f.write(f"\n**Source:** `{base_name}`\n")
            
            print(f"Created summary for {base_name}")
            
            # Remove old txt file if it exists
            old_txt = os.path.join(log_dir, f"{base_name}.summary.txt")
            if os.path.exists(old_txt):
                os.remove(old_txt)
                print(f"Removed old summary file: {old_txt}")

        else:
            print(f"No summary found in {os.path.basename(log_file)}")

if __name__ == "__main__":
    main()

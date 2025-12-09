#!/usr/bin/env python3
"""
Script to run Python scripts with proper environment setup and logging
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_script.py <script_name>")
        print("Example: python run_script.py optimize_prompt.py")
        sys.exit(1)
    
    script_name = sys.argv[1]
    
    # Check if script exists
    if not os.path.isfile(script_name):
        print(f"Error: Script '{script_name}' not found.")
        sys.exit(1)
    
    # Create a logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate a log file name based on the script and current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{os.path.splitext(script_name)[0]}_{timestamp}.log")
    
    print("")
    print(f"--- Running python {script_name} ---")
    print(f"--- Output will be logged to {log_file} ---")
    
    # Use python's -u flag for unbuffered output.
    # Redirect stderr (2) to stdout (1) so both are captured.
    # `tee` writes the output to the specified log file AND to the console.
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            [sys.executable, '-u', script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        process.wait()
    
    print("")
    print(f"--- Script finished. Log saved to: {log_file} ---")
    
    # Exit with the same code as the script
    sys.exit(process.returncode)

if __name__ == "__main__":
    main()

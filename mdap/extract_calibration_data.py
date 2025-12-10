#!/usr/bin/env python3
"""
Extract raw calibration data from MDAP logs for analysis.
Processes the most recent log file in the logs/ directory.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def find_most_recent_log() -> Optional[str]:
    """Find the most recent calibration log file in the logs directory."""
    # Check both project root logs and mdap/logs directories
    log_dirs = [Path("logs"), Path("mdap/logs")]
    
    for logs_dir in log_dirs:
        print(f"Checking directory: {logs_dir.absolute()}")
        if not logs_dir.exists():
            print(f"  Directory does not exist")
            continue
        
        # List all .log files for debugging
        all_logs = list(logs_dir.glob("*.log"))
        print(f"  Found {len(all_logs)} .log files:")
        for log in all_logs:
            print(f"    - {log.name}")
        
        # Look specifically for calibrate_hanoi_*.log files
        log_files = list(logs_dir.glob("calibrate_hanoi_*.log"))
        if log_files:
            print(f"  Found {len(log_files)} calibration log files")
            most_recent = max(log_files, key=lambda f: f.stat().st_mtime)
            print(f"  Using most recent: {most_recent.name}")
            return str(most_recent)
        else:
            print("  No calibrate_hanoi_*.log files found in this directory")
    
    print("No calibrate_hanoi log files found in any checked directory")
    return None

def extract_calibration_summary(log_content: str) -> Dict:
    """Extract the final summary from the calibration log."""
    summary = {}
    
    # Extract final p and k
    p_match = re.search(r'Estimated per-step success rate \(p\): ([\d.]+)', log_content)
    if p_match:
        summary['p_estimate'] = float(p_match.group(1))

    k_match = re.search(r'Recommended k_margin: (\d+)', log_content)
    if k_match:
        summary['k_margin'] = int(k_match.group(1))

    # Extract final counts
    count_match = re.search(r'Final count: successful_steps=(\d+), total_valid_steps=(\d+), red_flagged=(\d+)', log_content)
    if count_match:
        summary['successful_steps'] = int(count_match.group(1))
        summary['total_valid_steps'] = int(count_match.group(2))
        summary['red_flagged_steps'] = int(count_match.group(3))

    # Extract configuration
    model_match = re.search(r'API Call: model=([^,]+)', log_content)
    if model_match:
        summary['model'] = model_match.group(1)
        
    target_disks_match = re.search(r'Calculating optimal k_margin for target problem \((\d+) disks\)', log_content)
    if target_disks_match:
        summary['target_disks'] = int(target_disks_match.group(1))

    sample_steps_match = re.search(r'Sample Steps: (\d+)', log_content)
    if sample_steps_match:
        summary['sample_steps'] = int(sample_steps_match.group(1))

    return summary

def extract_step_details(log_content: str) -> List[Dict]:
    """Extract detailed information for each calibration step."""
    steps = []
    
    # Split log into lines for stateful parsing
    lines = log_content.split('\n')
    
    current_step = {}
    
    for line in lines:
        # Start of a new step
        state_match = re.search(r'Testing pre-generated state (\d+)/(\d+)', line)
        if state_match:
            if current_step: # Save previous step if it exists
                steps.append(current_step)
            step_num = int(state_match.group(1))
            current_step = {'step': step_num, 'status': 'PENDING'}

        # Extract optimal move
        optimal_match = re.search(r'Optimal move for state \d+: (\[.*?\])', line)
        if optimal_match and current_step:
            current_step['optimal_move'] = json.loads(optimal_match.group(1))

        # Extract raw LLM response
        if "RAW LLM RESPONSE" in line and current_step:
            current_step['raw_response_start_index'] = lines.index(line)
            
        # Extract parsed LLM response
        parsed_match = re.search(r'LLM Parsed Response: ({.*})', line)
        if parsed_match and current_step:
            try:
                current_step['llm_parsed_response'] = json.loads(parsed_match.group(1))
                current_step['llm_move'] = current_step['llm_parsed_response'].get("move", [])
            except json.JSONDecodeError:
                current_step['llm_parsed_response'] = parsed_match.group(1)
                current_step['llm_move'] = "PARSE_ERROR"

        # Extract voting result
        vote_match = re.search(r'Voting Result: winner=.* votes=(\d+)/(\d+)', line)
        if vote_match and current_step:
            current_step['winning_votes'] = int(vote_match.group(1))
            current_step['candidates_sampled'] = int(vote_match.group(2))

        # Determine success or failure
        success_match = re.search(r'State \d+: LLM move matches optimal move ✓', line)
        if success_match and current_step:
            current_step['status'] = 'SUCCESS'
        
        failure_match = re.search(r'State \d+: LLM move (\[.*?\]) != optimal move (\[.*?\]) ✗', line)
        if failure_match and current_step:
            current_step['status'] = 'FAILURE'
            current_step['llm_move'] = json.loads(failure_match.group(1))
            current_step['optimal_move'] = json.loads(failure_match.group(2))

        # Check for red flags
        red_flag_match = re.search(r'RED FLAG: (.*)', line)
        if red_flag_match and current_step:
            current_step.setdefault('red_flags', []).append(red_flag_match.group(1))

    # Add the last step
    if current_step:
        steps.append(current_step)
        
    # Now, extract the raw response text for each step
    for i, step in enumerate(steps):
        if 'raw_response_start_index' in step:
            start_idx = step['raw_response_start_index']
            # Find the end of the response block
            end_idx = start_idx + 1
            while end_idx < len(lines) and not lines[end_idx].startswith('---'):
                end_idx += 1
            raw_response_block = "\n".join(lines[start_idx+1 : end_idx])
            step['raw_response'] = raw_response_block.strip()
            del step['raw_response_start_index']

    return steps

def generate_analysis_markdown(summary: Dict, steps: List[Dict]) -> str:
    """Generate a markdown report for easy analysis."""
    report = []
    report.append("# Calibration Data Analysis\n")
    
    # Configuration Section
    report.append("## Configuration")
    report.append(f"- **Model:** `{summary.get('model', 'N/A')}`")
    report.append(f"- **Target Problem Size:** {summary.get('target_disks', 'N/A')} disks")
    report.append(f"- **Calibration Sample Size:** {summary.get('sample_steps', 'N/A')} steps")
    report.append("\n---\n")

    # Summary Section
    report.append("## Summary")
    report.append(f"- **Estimated per-step success rate (p):** `{summary.get('p_estimate', 'N/A'):.4f}`")
    report.append(f"- **Recommended k_margin:** `{summary.get('k_margin', 'N/A')}`")
    report.append(f"- **Total Steps Tested:** {summary.get('total_valid_steps', 0)}")
    report.append(f"- **Successful Steps:** {summary.get('successful_steps', 0)}")
    report.append(f"- **Failed Steps:** {summary.get('total_valid_steps', 0) - summary.get('successful_steps', 0)}")
    report.append(f"- **Red-Flagged Steps:** {summary.get('red_flagged_steps', 0)}")
    report.append("\n---\n")

    # Step-by-Step Breakdown
    report.append("## Step-by-Step Breakdown\n")
    
    failed_steps = [s for s in steps if s.get('status') == 'FAILURE']
    successful_steps = [s for s in steps if s.get('status') == 'SUCCESS']

    if failed_steps:
        report.append("### ❌ Failed Steps")
        for step in failed_steps:
            report.append(f"\n#### Step {step['step']}: FAILURE")
            report.append(f"- **Optimal Move:** `{step.get('optimal_move', 'N/A')}`")
            report.append(f"- **LLM Move:** `{step.get('llm_move', 'N/A')}`")
            report.append(f"- **Voting:** Winner found with {step.get('winning_votes', 'N/A')} votes after {step.get('candidates_sampled', 'N/A')} candidates.")
            if 'red_flags' in step:
                report.append(f"- **Red Flags:**")
                for flag in step['red_flags']:
                    report.append(f"  - {flag}")
            report.append("- **Raw LLM Response:**")
            report.append("```")
            report.append(step.get('raw_response', 'N/A'))
            report.append("```")
        report.append("\n---\n")

    if successful_steps:
        report.append("### ✅ Successful Steps")
        # Show a few examples, not all, to keep it concise
        for step in successful_steps[:5]:
            report.append(f"\n#### Step {step['step']}: SUCCESS")
            report.append(f"- **Optimal Move:** `{step.get('optimal_move', 'N/A')}`")
            report.append(f"- **LLM Move:** `{step.get('llm_move', 'N/A')}`")
            report.append(f"- **Voting:** Winner found with {step.get('winning_votes', 'N/A')} votes after {step.get('candidates_sampled', 'N/A')} candidates.")
        if len(successful_steps) > 5:
            report.append(f"\n... and {len(successful_steps) - 5} other successful steps.")
            
    return "\n".join(report)

def main():
    """Main execution function."""
    log_file = find_most_recent_log()
    if not log_file:
        return
    
    print(f"Processing calibration log: {log_file}")
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Check if this is actually a calibration log by looking for key phrases
    if "Estimating per-step success rate" not in content and "Calibration Result" not in content:
        print("Warning: This does not appear to be a calibration log. Exiting.")
        return

    print("Extracting calibration summary...")
    summary = extract_calibration_summary(content)
    print(f"Summary extracted: {bool(summary)}")
    
    print("Extracting step details...")
    steps = extract_step_details(content)
    print(f"Steps extracted: {len(steps) if steps else 0}")
    
    if not summary:
        print("ERROR: Could not extract calibration summary from the log.")
        print("\nDebug: Checking for key patterns in log:")
        print(f"  - 'Estimated per-step success rate': {'Found' if 'Estimated per-step success rate' in content else 'NOT FOUND'}")
        print(f"  - 'Recommended k_margin': {'Found' if 'Recommended k_margin' in content else 'NOT FOUND'}")
        print(f"  - 'Final count:': {'Found' if 'Final count:' in content else 'NOT FOUND'}")
        return
    
    if not steps:
        print("ERROR: Could not extract step details from the log.")
        print("\nDebug: Checking for step patterns in log:")
        print(f"  - 'Testing pre-generated state': {'Found' if 'Testing pre-generated state' in content else 'NOT FOUND'}")
        print(f"  - 'Optimal move for state': {'Found' if 'Optimal move for state' in content else 'NOT FOUND'}")
        print(f"  - 'RAW LLM RESPONSE': {'Found' if 'RAW LLM RESPONSE' in content else 'NOT FOUND'}")
        return

    report = generate_analysis_markdown(summary, steps)
    
    # Save to a file
    output_file = f"docs/analysis/calibration_data_analysis.md"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nCalibration analysis report saved to: {output_file}\n")
    print("="*50)
    print(report)

if __name__ == "__main__":
    main()

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
    # Temporarily hardcode the specific log file for analysis
    specific_log = Path("mdap/logs/calibrate_hanoi_20251210_054304.log")
    if specific_log.exists():
        return str(specific_log)

    # Fallback to the original logic if the specific file is not found
    # Check both project root logs and mdap/logs directories
    log_dirs = [Path("logs"), Path("mdap/logs")]
    
    for logs_dir in log_dirs:
        if not logs_dir.exists():
            continue
        
        # Look specifically for calibrate_hanoi_*.log files
        log_files = list(logs_dir.glob("calibrate_hanoi_*.log"))
        if log_files:
            most_recent = max(log_files, key=lambda f: f.stat().st_mtime)
            return str(most_recent)
    
    print("No calibrate_hanoi log files found")
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
    step_counter = 0
    
    for i, line in enumerate(lines):
        # Start of a new step - look for estimation loop
        state_match = re.search(r'Estimation loop iteration (\d+)/(\d+)', line)
        if state_match:
            if current_step: # Save previous step if it exists
                steps.append(current_step)
            step_counter += 1
            current_step = {'step': step_counter, 'status': 'PENDING'}

        # Extract optimal move
        optimal_match = re.search(r'Optimal move for step (\d+): (\[.*?\])', line)
        if optimal_match and current_step:
            current_step['optimal_move'] = json.loads(optimal_match.group(2))

        # Extract raw LLM response
        if "RAW LLM RESPONSE" in line and current_step:
            # Find the end of the response block
            end_idx = i + 1
            while end_idx < len(lines) and not lines[end_idx].startswith('---'):
                end_idx += 1
            raw_response_block = "\n".join(lines[i+1 : end_idx])
            current_step['raw_response'] = raw_response_block.strip()
            
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
        success_match = re.search(r'Step (\d+): LLM move matches optimal move ✓', line)
        if success_match and current_step:
            current_step['status'] = 'SUCCESS'
        
        failure_match = re.search(r'Step (\d+): LLM move (\[.*?\]) != optimal move (\[.*?\]) ✗', line)
        if failure_match and current_step:
            current_step['status'] = 'FAILURE'
            current_step['llm_move'] = json.loads(failure_match.group(2))
            current_step['optimal_move'] = json.loads(failure_match.group(3))

        # Check for red flags
        red_flag_match = re.search(r'RED FLAG: (.*)', line)
        if red_flag_match and current_step:
            current_step.setdefault('red_flags', []).append(red_flag_match.group(1))

    # Add the last step
    if current_step:
        steps.append(current_step)

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
    p_val = summary.get('p_estimate', 'N/A')
    if isinstance(p_val, (int, float)):
        p_formatted = f"{p_val:.4f}"
    else:
        p_formatted = str(p_val)
    report.append(f"- **Estimated per-step success rate (p):** `{p_formatted}`")
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
        return
    
    # If no step details found, it's likely using cached calibration
    if not steps:
        if "Loaded calibration cache" in content:
            print("\nNote: This calibration used cached states.")
            print("Step-by-step LLM interactions are not available in this log.")
            print("To see detailed step analysis, re-run calibration with --regenerate_cache")
        else:
            print("\nNote: Step details not found in this log file.")
            print("The detailed LLM interactions may be in a separate harness log file.")
        
        # Generate a summary report even without step details
        report = generate_summary_report(summary)
    else:
        report = generate_analysis_markdown(summary, steps)
    
    # Save to a file
    output_file = f"docs/analysis/calibration_data_analysis.md"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nCalibration analysis report saved to: {output_file}\n")
    print("="*50)
    print(report)

def generate_summary_report(summary: Dict) -> str:
    """Generate a summary report when step details aren't available."""
    report = []
    report.append("# Calibration Summary\n")
    
    # Configuration Section
    report.append("## Configuration")
    report.append(f"- **Model:** `{summary.get('model', 'N/A')}`")
    report.append(f"- **Target Problem Size:** {summary.get('target_disks', 'N/A')} disks")
    report.append(f"- **Calibration Sample Size:** {summary.get('sample_steps', 'N/A')} steps")
    report.append("\n---\n")

    # Results Section
    report.append("## Results")
    p_val = summary.get('p_estimate', 'N/A')
    if isinstance(p_val, (int, float)):
        p_formatted = f"{p_val:.4f}"
    else:
        p_formatted = str(p_val)
    report.append(f"- **Estimated per-step success rate (p):** `{p_formatted}`")
    report.append(f"- **Recommended k_margin:** `{summary.get('k_margin', 'N/A')}`")
    report.append(f"- **Total Steps Tested:** {summary.get('total_valid_steps', 0)}")
    report.append(f"- **Successful Steps:** {summary.get('successful_steps', 0)}")
    report.append(f"- **Failed Steps:** {summary.get('total_valid_steps', 0) - summary.get('successful_steps', 0)}")
    report.append(f"- **Red-Flagged Steps:** {summary.get('red_flagged_steps', 0)}")
    report.append("\n---\n")

    # Analysis Section
    p = summary.get('p_estimate', 0)
    k = summary.get('k_margin', 0)
    
    report.append("## Analysis")
    if p > 0.95:
        report.append(f"✅ **Excellent performance** (p={p:.3f}). The model is highly reliable.")
    elif p > 0.8:
        report.append(f"✅ **Good performance** (p={p:.3f}). The model is reliable with moderate voting.")
    elif p > 0.6:
        report.append(f"⚠️ **Moderate performance** (p={p:.3f}). Requires significant voting (k={k}).")
    elif p > 0.5:
        report.append(f"❌ **Poor performance** (p={p:.3f}). Barely better than random, high k needed.")
    else:
        report.append(f"❌ **Very poor performance** (p={p:.3f}). Model may not be suitable for this task.")
    
    report.append("\n### Recommendations")
    if p < 0.6:
        report.append("- Consider using a more capable model")
        report.append("- Check if red-flagging is too strict")
        report.append("- Review prompt engineering")
    elif k > 10:
        report.append(f"- High k_margin ({k}) will result in significant API costs")
        report.append("- Consider optimizing for better per-step accuracy")
    
    return "\n".join(report)

if __name__ == "__main__":
    main()

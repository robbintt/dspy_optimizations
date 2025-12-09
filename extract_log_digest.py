#!/usr/bin/env python3
"""
Extract token-efficient digest of LLM responses and paper parameters from MDAP logs.
Processes the most recent log file in the logs/ directory.
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def find_most_recent_log() -> Optional[str]:
    """Find the most recent log file in the logs directory"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("No logs directory found")
        return None
    
    # Get all .log files
    log_files = list(logs_dir.glob("*.log"))
    if not log_files:
        print("No log files found")
        return None
    
    # Sort by modification time
    most_recent = max(log_files, key=lambda f: f.stat().st_mtime)
    return str(most_recent)

def extract_llm_responses(log_content: str) -> List[Dict]:
    """Extract LLM parsed responses from log"""
    responses = []
    
    # Pattern to match LLM Parsed Response entries
    pattern = r'LLM Parsed Response: ({.*?})'
    matches = re.findall(pattern, log_content)
    
    for i, match in enumerate(matches, 1):
        try:
            response_data = json.loads(match)
            responses.append({
                "step": i,
                "move": response_data.get("move", []),
                "predicted_state": response_data.get("predicted_state", {})
            })
        except json.JSONDecodeError:
            # Skip invalid JSON
            continue
    
    return responses

def extract_voting_info(log_content: str) -> List[Dict]:
    """Extract voting information for each step"""
    voting_info = []
    
    # Extract candidate attempts and winner info
    candidate_pattern = r'Attempting to get candidate (\d+)/10'
    winner_pattern = r'Winner found with (\d+) votes \(reached k_margin\)'
    
    candidates = []
    current_step = 1
    
    for line in log_content.split('\n'):
        if 'Attempting to get candidate' in line:
            match = re.search(candidate_pattern, line)
            if match:
                candidates.append(int(match.group(1)))
        elif 'Winner found' in line:
            match = re.search(winner_pattern, line)
            if match:
                voting_info.append({
                    "step": current_step,
                    "candidates_sampled": len(candidates),
                    "winning_votes": int(match.group(1)),
                    "k_margin": 3  # From config
                })
                candidates = []
                current_step += 1
    
    return voting_info

def extract_paper_parameters(log_content: str) -> Dict:
    """Extract MAKER framework parameters from the log"""
    params = {}
    
    # Extract model info
    model_match = re.search(r'LiteLLM completion\(\) model= ([^;]+)', log_content)
    if model_match:
        params["model"] = model_match.group(1)
    
    # Extract k_margin
    k_margin_match = re.search(r'first-to-ahead-by-K with k_margin=(\d+)', log_content)
    if k_margin_match:
        params["k_margin"] = int(k_margin_match.group(1))
    
    # Extract max_candidates
    max_candidates_match = re.search(r'max_candidates=(\d+)', log_content)
    if max_candidates_match:
        params["max_candidates"] = int(max_candidates_match.group(1))
    
    # Extract temperature
    temp_match = re.search(r'temperature=([\d.]+)', log_content)
    if temp_match:
        params["temperature"] = float(temp_match.group(1))
    
    # Extract problem info
    problem_match = re.search(r'Attempting to solve (\d+)-disk Towers of Hanoi', log_content)
    if problem_match:
        params["num_disks"] = int(problem_match.group(1))
        params["total_steps_optimal"] = 2**int(problem_match.group(1)) - 1
    
    # Extract execution metrics
    steps_match = re.search(r'MDAP execution completed in (\d+) steps', log_content)
    if steps_match:
        params["actual_steps"] = int(steps_match.group(1))
    
    moves_match = re.search(r'Solution completed in (\d+) moves', log_content)
    if moves_match:
        params["total_moves"] = int(moves_match.group(1))
    
    return params

def extract_state_transitions(log_content: str) -> List[Dict]:
    """Extract state transitions for each step"""
    transitions = []
    
    # Pattern to match state updates
    state_pattern = r'State updated successfully: HanoiState\(pegs=(\{[^}]+\}),'
    matches = re.findall(state_pattern, log_content)
    
    for i, match in enumerate(matches, 1):
        try:
            pegs_dict = eval(match)  # Convert string representation to dict
            transitions.append({
                "step": i,
                "pegs": pegs_dict
            })
        except:
            continue
    
    return transitions

def generate_digest(responses: List[Dict], voting: List[Dict], 
                   params: Dict, transitions: List[Dict]) -> str:
    """Generate token-efficient digest"""
    digest = []
    
    # Header with key parameters
    digest.append("# MDAP Execution Digest")
    digest.append(f"Model: {params.get('model', 'N/A')}")
    digest.append(f"Problem: {params.get('num_disks', 'N/A')}-disk Hanoi")
    digest.append(f"Params: k={params.get('k_margin', 'N/A')}, max_cand={params.get('max_candidates', 'N/A')}, temp={params.get('temperature', 'N/A')}")
    digest.append("")
    
    # Step-by-step summary
    digest.append("## Step Summary")
    for i, (resp, vote, trans) in enumerate(zip(responses, voting, transitions), 1):
        digest.append(f"Step {i}:")
        digest.append(f"  Move: {resp['move']}")
        digest.append(f"  Votes: {vote['winning_votes']}/{vote['candidates_sampled']}")
        digest.append(f"  State: {trans['pegs']}")
        digest.append("")
    
    # Performance metrics
    digest.append("## Performance")
    digest.append(f"Optimal steps: {params.get('total_steps_optimal', 'N/A')}")
    digest.append(f"Actual steps: {params.get('actual_steps', 'N/A')}")
    digest.append(f"Total API calls: {sum(v['candidates_sampled'] for v in voting)}")
    digest.append(f"Efficiency: {params.get('actual_steps', 'N/A')}/{params.get('total_steps_optimal', 'N/A')} = {params.get('actual_steps', 1)/params.get('total_steps_optimal', 1):.2f}")
    
    return "\n".join(digest)

def main():
    """Main execution function"""
    # Find most recent log
    log_file = find_most_recent_log()
    if not log_file:
        return
    
    print(f"Processing log: {log_file}")
    
    # Read log content
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract information
    responses = extract_llm_responses(content)
    voting = extract_voting_info(content)
    params = extract_paper_parameters(content)
    transitions = extract_state_transitions(content)
    
    # Generate digest
    digest = generate_digest(responses, voting, params, transitions)
    
    # Save digest
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest_file = f"docs/analysis/log_digest_{timestamp}.md"
    
    os.makedirs(os.path.dirname(digest_file), exist_ok=True)
    with open(digest_file, 'w') as f:
        f.write(digest)
    
    print(f"Digest saved to: {digest_file}")
    print("\n" + "="*50)
    print(digest)

if __name__ == "__main__":
    main()

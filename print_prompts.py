import json
import sys

def print_prompts_from_json(file_path):
    """
    Loads a GEPA results JSON and prints the evolved prompts for each component.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'")
        sys.exit(1)

    candidates = data.get("candidates")
    if not candidates or not isinstance(candidates, list):
        print("Error: 'candidates' key not found or is not a list in the JSON file.")
        sys.exit(1)

    best_idx = data.get("best_idx", 0)
    if not isinstance(best_idx, int) or best_idx >= len(candidates):
        print(f"Warning: Invalid 'best_idx' ({best_idx}). Defaulting to index 0.")
        best_idx = 0

    print(f"--- Evolved Prompts from GEPA Run ---")
    print(f"Source File: {file_path}")
    print(f"Total Candidates Explored: {len(candidates)}")
    print(f"Best Performer (Candidate Index: {best_idx})")
    print("=" * 40)

    best_candidate = candidates[best_idx]

    if not isinstance(best_candidate, dict):
        print("Error: Best candidate is not a dictionary.")
        sys.exit(1)
        
    for component_name, instructions in best_candidate.items():
        print(f"\n[ Component: {component_name} ]")
        print("-" * (len(component_name) + 14))
        print(instructions)
        print("-" * (len(component_name) + 14))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_prompts.py <path_to_gepa_complete.json>")
        sys.exit(1)
        
    json_file_path = sys.argv[1]
    print_prompts_from_json(json_file_path)

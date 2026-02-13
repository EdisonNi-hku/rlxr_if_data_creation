import json
import argparse
from collections import defaultdict

def calculate_metrics(input_file):
    total_score = 0
    count = 0
    reward_totals = defaultdict(float)
    reward_counts = defaultdict(int)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Update score
            if 'avg_score' in data:
                total_score += data['avg_score']
                count += 1
            
            # Update rewards
            for key, value in data.items():
                if key.startswith('avg_reward_'):
                    reward_totals[key] += value
                    reward_counts[key] += 1

    if count == 0:
        print("No data found.")
        return

    avg_score = total_score / count
    print(f"Total entries: {count}")
    print(f"Average Score: {avg_score:.4f}")
    
    print("\nAverage Rewards by Model:")
    for key in sorted(reward_totals.keys()):
        avg_reward = reward_totals[key] / reward_counts[key]
        # Clean up the key for display if possible, or just print it
        model_name = key.replace('avg_reward_', '')
        print(f"  {model_name}: {avg_reward:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average score and rewards from a JSONL file.")
    parser.add_argument("input_file", help="Path to the JSONL file.")
    args = parser.parse_args()
    
    calculate_metrics(args.input_file)

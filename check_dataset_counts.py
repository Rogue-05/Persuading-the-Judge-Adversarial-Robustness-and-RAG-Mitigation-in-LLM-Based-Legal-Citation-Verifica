import json
from collections import defaultdict

def main():
    try:
        with open("legal_dataset.json", "r") as f:
            data = json.load(f)
            
        items = data.get("items", [])
        if not items:
            print("No items found in legal_dataset.json.")
            return

        # Structure: counts[topic][category] = count
        counts = defaultdict(lambda: defaultdict(int))
        
        for item in items:
            topic = item.get("topic", "Unknown Topic")
            category = item.get("category", "Unknown Category")
            counts[topic][category] += 1
            
        print("=== Dataset Summary by Topic ===\n")
        
        total_valid = 0
        total_wrong = 0
        total_fab = 0
        
        for topic, category_counts in counts.items():
            valid = category_counts.get("valid", 0)
            wrong = category_counts.get("real_wrong_content", 0)
            fab = category_counts.get("fabricated", 0)
            
            total_valid += valid
            total_wrong += wrong
            total_fab += fab
            
            print(f"Topic: {topic}")
            print(f"  - Valid cases:      {valid}")
            print(f"  - Real Wrong cases: {wrong}")
            print(f"  - Fabricated cases: {fab}")
            print("-" * 30)
            
        print("\n=== Grand Totals ===")
        print(f"Total Topics:     {len(counts)}")
        print(f"Total Valid:      {total_valid}")
        print(f"Total Real Wrong: {total_wrong}")
        print(f"Total Fabricated: {total_fab}")
        print(f"Total Items:      {len(items)}")

    except FileNotFoundError:
        print("Error: legal_dataset.json not found in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

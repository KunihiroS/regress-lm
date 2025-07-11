import os
import glob
import yaml
import re
import math

def find_leaks():
    project_root = '/home/kunihiros/dev/regress-lm'
    train_dir = os.path.join(project_root, 'test/cleaned_data/train')
    
    files_with_leaks = []

    for f_path in sorted(glob.glob(os.path.join(train_dir, '*.yml'))):
        with open(f_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
                news = data.get('news', '')
                change_rate = data.get('change_rate')

                if change_rate is None:
                    continue

                # Regex to find numbers followed by '%'
                matches = re.findall(r'(\d+\.?\d*)\s*%', news)
                
                for match in matches:
                    try:
                        # Convert matched percentage string to a float ratio
                        news_rate = float(match) / 100.0
                        
                        # Compare absolute values to handle 'decline' vs 'rise'
                        if math.isclose(abs(news_rate), abs(change_rate), rel_tol=1e-5):
                            files_with_leaks.append({
                                'file': os.path.basename(f_path),
                                'news': news,
                                'change_rate': change_rate,
                                'leaked_value': f"{match}%"
                            })
                            break 
                    except (ValueError, TypeError):
                        continue
            except (yaml.YAMLError, AttributeError) as e:
                print(f"Could not process file {f_path}: {e}")
                continue
    
    if not files_with_leaks:
        print("No direct leaks found where the percentage in 'news' matches 'change_rate'.")
        return

    print("Found files where the percentage in 'news' directly matches 'change_rate':\n")
    for item in files_with_leaks:
        print(f"File: {item['file']}")
        print(f"  News: \"{item['news']}\"")
        print(f"  Change Rate: {item['change_rate']}")
        print(f"  Matching Leaked Value: {item['leaked_value']}")
        print("-" * 20)

if __name__ == '__main__':
    find_leaks()

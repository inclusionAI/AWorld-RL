#!/usr/bin/env python3
# Evaluator for reservation Query 6

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class FavoritesHTMLParser(HTMLParser):
    """Parse HTML to extract favorites page information."""
    def __init__(self):
        super().__init__()
        self.in_favorites_subtitle = False
        self.in_favorites_grid = False
        self.in_favorite_card = False
        self.in_favorite_name = False
        self.saved_count = None
        self.restaurant_names = []
        self.current_name = ""
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == "p" and attrs_dict.get("class") == "favorites-subtitle":
            self.in_favorites_subtitle = True
        elif tag == "div" and attrs_dict.get("class") == "favorites-grid":
            self.in_favorites_grid = True
        elif tag == "div" and "favorite-card" in attrs_dict.get("class", ""):
            self.in_favorite_card = True
        elif tag == "h3" and "favorite-name" in attrs_dict.get("class", ""):
            self.in_favorite_name = True
            
    def handle_endtag(self, tag):
        if tag == "p" and self.in_favorites_subtitle:
            self.in_favorites_subtitle = False
        elif tag == "div" and self.in_favorites_grid:
            self.in_favorites_grid = False
        elif tag == "div" and self.in_favorite_card:
            self.in_favorite_card = False
        elif tag == "h3" and self.in_favorite_name:
            self.in_favorite_name = False
            if self.current_name:
                self.restaurant_names.append(self.current_name.strip())
                self.current_name = ""
        
    def handle_data(self, data):
        if self.in_favorites_subtitle:
            match = re.search(r'\((\d+)\)', data)
            if match:
                self.saved_count = int(match.group(1))
        elif self.in_favorite_name:
            self.current_name += data


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        self.checkpoints = {
            "cp1_navigated_to_favorites": False,
            "cp2_initial_favorites_count": False,
            "cp3_removed_first_restaurant": False,
            "cp4_final_favorites_count": False,
        }
        
        self.issues = []
        self.details = {}
    
    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        final_files = list(self.result_dir.glob("final_*_raw.html"))
        if final_files:
            return final_files[0]
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        return step_files[-1] if step_files else None
    
    def parse_html_favorites(self, html_path: Path) -> Dict:
        """Parse HTML file to extract favorites information."""
        parser = FavoritesHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        
        return {
            'saved_count': parser.saved_count,
            'restaurant_names': parser.restaurant_names
        }
    
    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data."""
        if not self.traj_file.exists():
            return []
        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                trajectory.append(json.loads(line))
        return trajectory
    
    def load_result_json(self) -> Dict:
        """Load result.json."""
        with open(self.result_file, 'r') as f:
            return json.load(f)
    
    def check_checkpoint_1(self) -> bool:
        """Check if agent navigated to favorites page."""
        try:
            step_1_files = list(self.result_dir.glob("step_1_*_raw.html"))
            if not step_1_files:
                self.issues.append("No step_1 HTML file found")
                return False
            
            with open(step_1_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
                if 'favorites-page' in html_content or 'My Favorites' in html_content:
                    self.details['navigated_to_favorites'] = True
                    return True
                else:
                    self.issues.append("Did not navigate to favorites page")
                    return False
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if initial state had 2 saved restaurants."""
        try:
            step_1_files = list(self.result_dir.glob("step_1_*_raw.html"))
            if not step_1_files:
                self.issues.append("No step_1 HTML file found")
                return False
            
            favorites_data = self.parse_html_favorites(step_1_files[0])
            self.details['initial_count'] = favorites_data['saved_count']
            self.details['initial_restaurants'] = favorites_data['restaurant_names']
            
            if favorites_data['saved_count'] == 2:
                return True
            else:
                self.issues.append(f"Initial favorites count was {favorites_data['saved_count']}, expected 2")
                return False
        except Exception as e:
            self.issues.append(f"Error checking initial count: {str(e)}")
            return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if the first restaurant was removed."""
        try:
            step_1_files = list(self.result_dir.glob("step_1_*_raw.html"))
            if not step_1_files:
                self.issues.append("No step_1 HTML file found")
                return False
            
            initial_data = self.parse_html_favorites(step_1_files[0])
            
            if not initial_data['restaurant_names']:
                self.issues.append("No restaurants found in initial state")
                return False
            
            first_restaurant = initial_data['restaurant_names'][0]
            self.details['first_restaurant'] = first_restaurant
            
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            final_data = self.parse_html_favorites(final_html)
            self.details['final_restaurants'] = final_data['restaurant_names']
            
            if first_restaurant not in final_data['restaurant_names']:
                self.details['first_restaurant_removed'] = True
                return True
            else:
                self.issues.append(f"First restaurant '{first_restaurant}' was not removed")
                return False
        except Exception as e:
            self.issues.append(f"Error checking restaurant removal: {str(e)}")
            return False
    
    def check_checkpoint_4(self) -> bool:
        """Check if final state has 1 saved restaurant."""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            final_data = self.parse_html_favorites(final_html)
            self.details['final_count'] = final_data['saved_count']
            
            if final_data['saved_count'] == 1:
                return True
            else:
                self.issues.append(f"Final favorites count was {final_data['saved_count']}, expected 1")
                return False
        except Exception as e:
            self.issues.append(f"Error checking final count: {str(e)}")
            return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            self.checkpoints['cp1_navigated_to_favorites'] = self.check_checkpoint_1()
            self.checkpoints['cp2_initial_favorites_count'] = self.check_checkpoint_2()
            self.checkpoints['cp3_removed_first_restaurant'] = self.check_checkpoint_3()
            self.checkpoints['cp4_final_favorites_count'] = self.check_checkpoint_4()
            
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 6,
                'query': result_data.get('query', ''),
                'overall_success': overall_success,
                'success_rate': success_rate,
                'checkpoints': self.checkpoints,
                'checkpoints_passed': f"{passed_count}/{total_count}",
                'issues': self.issues,
                'details': self.details,
                'agent_claimed_success': result_data.get('success', False),
                'execution_time': result_data.get('execution_time', 0),
                'total_steps': result_data.get('steps', 0)
            }
        except Exception as e:
            return {
                'query_id': 6,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query6.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

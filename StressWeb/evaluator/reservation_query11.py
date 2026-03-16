#!/usr/bin/env python3
# Evaluator for reservation Query 11

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract restaurant information from favorites page."""

    def __init__(self):
        super().__init__()
        self.in_favorites_page = False
        self.in_restaurant_name = False
        self.in_restaurant_features = False
        self.current_tag = None
        self.restaurant_names = []
        self.current_restaurant = None
        self.restaurants_with_features = {}
        self.page_class = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're on favorites page
        if tag == 'div' and 'class' in attrs_dict:
            if 'favorites-page' in attrs_dict['class']:
                self.in_favorites_page = True
            elif 'search-results-page' in attrs_dict['class']:
                self.in_favorites_page = False
                
        # Check for restaurant name
        if tag == 'h3' and 'class' in attrs_dict:
            if 'restaurant-name' in attrs_dict['class']:
                self.in_restaurant_name = True
                
        # Check for features section
        if tag == 'div' and 'class' in attrs_dict:
            if 'restaurant-features' in attrs_dict['class']:
                self.in_restaurant_features = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        # Capture restaurant names
        if self.in_restaurant_name:
            self.restaurant_names.append(data)
            self.current_restaurant = data
            self.restaurants_with_features[data] = []
            self.in_restaurant_name = False
            
        # Capture features
        if self.in_restaurant_features and self.current_restaurant:
            if '🌿' in data or 'Outdoor Seating' in data:
                if 'outdoor_seating' not in self.restaurants_with_features[self.current_restaurant]:
                    self.restaurants_with_features[self.current_restaurant].append('outdoor_seating')

    def handle_endtag(self, tag):
        if tag == 'div':
            self.in_restaurant_features = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for this task
        self.checkpoints = {
            "cp1_searched_asian_fusion": False,
            "cp2_filtered_outdoor_seating": False,
            "cp3_saved_restaurants": False,
            "cp4_viewed_favorites_page": False,
        }

        self.issues = []
        self.details = {}

    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        final_files = list(self.result_dir.glob("final_*_raw.html"))
        if final_files:
            return final_files[0]
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        if step_files:
            return step_files[-1]
        return None

    def find_step_html(self, step_pattern: str) -> List[Path]:
        """Find intermediate step HTML files."""
        return sorted(self.result_dir.glob(f"{step_pattern}_raw.html"))

    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data from traj.jsonl."""
        if not self.traj_file.exists():
            return []

        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                trajectory.append(json.loads(line))
        return trajectory

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Searched for Asian Fusion restaurants"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Look for TYPE action with "Asian Fusion"
            for action in actions:
                if action.get('action', {}).get('action_type') == 'TYPE':
                    text = action.get('action', {}).get('parameters', {}).get('text', '')
                    if 'Asian Fusion' in text:
                        self.details['searched_text'] = text
                        return True
                        
            self.issues.append("Did not search for 'Asian Fusion' restaurants")
            return False
        except Exception as e:
            self.issues.append(f"Error checking search: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Filtered by outdoor seating"""
        try:
            # Check intermediate steps to see if outdoor seating filter was applied
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Look for checked outdoor seating checkbox
                    if 'type="checkbox"' in content and 'Outdoor Seating' in content:
                        # Check if checkbox is checked (has checked attribute or style)
                        if 'checked' in content or ('input type="checkbox" style' in content and 'Outdoor Seating' in content):
                            self.details['outdoor_seating_filtered'] = True
                            return True
                            
            # Also check trajectory for click on outdoor seating
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            for action in actions:
                reasoning = action.get('action', {}).get('reasoning', '')
                if 'outdoor seating' in reasoning.lower() and 'filter' in reasoning.lower():
                    return True
                if 'outdoor seating' in reasoning.lower() and 'checkbox' in reasoning.lower():
                    return True
                    
            self.issues.append("Did not filter by outdoor seating")
            return False
        except Exception as e:
            self.issues.append(f"Error checking filter: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Saved Asian Fusion restaurants with outdoor seating"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Count save/favorite actions
            save_count = 0
            for action in actions:
                reasoning = action.get('action', {}).get('reasoning', '').lower()
                if 'save' in reasoning or 'favorite' in reasoning or 'heart' in reasoning:
                    if action.get('result', {}).get('success'):
                        save_count += 1
                        
            if save_count >= 2:  # At least 2 restaurants should be saved
                self.details['restaurants_saved'] = save_count
                return True
            else:
                self.issues.append(f"Only saved {save_count} restaurants, expected at least 2")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking saved restaurants: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Viewed favorites page"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
                
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if on favorites page
            if 'favorites-page' in content or 'My Favorite Restaurants' in content:
                self.details['on_favorites_page'] = True
                
                # Parse to count restaurants
                parser = TaskHTMLParser()
                parser.feed(content)
                
                if parser.in_favorites_page:
                    restaurant_count = len(parser.restaurant_names)
                    self.details['favorites_count'] = restaurant_count
                    self.details['favorite_restaurants'] = parser.restaurant_names
                    
                    if restaurant_count >= 2:
                        return True
                    else:
                        self.issues.append(f"Only {restaurant_count} restaurants in favorites, expected at least 2")
                        return False
                else:
                    # Fallback: just check we're on the page
                    return True
            else:
                self.issues.append("Not on favorites page in final state")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking favorites page: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_searched_asian_fusion'] = self.check_checkpoint_1()
            self.checkpoints['cp2_filtered_outdoor_seating'] = self.check_checkpoint_2()
            self.checkpoints['cp3_saved_restaurants'] = self.check_checkpoint_3()
            self.checkpoints['cp4_viewed_favorites_page'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 11,
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
                'query_id': 11,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query11.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

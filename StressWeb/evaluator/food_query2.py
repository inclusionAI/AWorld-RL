#!/usr/bin/env python3
# Evaluator for food Query 2

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class RestaurantCardParser(HTMLParser):
    """Parse HTML to extract restaurant cards and their favorite status."""

    def __init__(self):
        super().__init__()
        self.restaurants = []
        self.current_restaurant = None
        self.in_result_card = False
        self.in_result_name = False
        self.in_detail_item = False
        self.detail_item_count = 0
        self.current_detail_text = ""
        self.in_fastest_badge = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and attrs_dict.get('class') == 'result-card':
            self.in_result_card = True
            self.current_restaurant = {
                'name': '',
                'delivery_time': '',
                'is_favorited': False,
                'has_fastest_badge': False
            }
            
        elif self.in_result_card:
            if tag == 'button' and 'favorite-btn' in attrs_dict.get('class', ''):
                # Check if favorited class is present
                if 'favorited' in attrs_dict.get('class', ''):
                    self.current_restaurant['is_favorited'] = True
                    
            elif tag == 'h3' and attrs_dict.get('class') == 'result-name':
                self.in_result_name = True
                
            elif tag == 'div' and attrs_dict.get('class') == 'fastest-badge':
                self.in_fastest_badge = True
                self.current_restaurant['has_fastest_badge'] = True
                
            elif tag == 'div' and attrs_dict.get('class') == 'detail-item':
                self.in_detail_item = True
                self.detail_item_count += 1
                self.current_detail_text = ""

    def handle_data(self, data):
        if self.in_result_name:
            self.current_restaurant['name'] = data.strip()
        elif self.in_detail_item:
            text = data.strip()
            if text:
                self.current_detail_text += text

    def handle_endtag(self, tag):
        if tag == 'h3' and self.in_result_name:
            self.in_result_name = False
            
        elif tag == 'div' and self.in_detail_item:
            self.in_detail_item = False
            # First detail-item is usually delivery time
            if self.detail_item_count == 1 and 'min' in self.current_detail_text:
                self.current_restaurant['delivery_time'] = self.current_detail_text
                
        elif tag == 'div' and self.in_result_card:
            # Check if this is the end of result-card
            if self.current_restaurant and self.current_restaurant.get('name'):
                self.restaurants.append(self.current_restaurant)
                self.current_restaurant = None
                self.in_result_card = False
                self.detail_item_count = 0


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        self.checkpoints = {
            "cp1_italian_search_executed": False,
            "cp2_sorted_by_delivery_time": False,
            "cp3_fastest_restaurant_identified": False,
            "cp4_fastest_restaurant_favorited": False,
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

    def parse_restaurants_from_html(self, html_path: Path) -> List[Dict]:
        """Parse restaurant cards from HTML file."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = RestaurantCardParser()
        parser.feed(html_content)
        return parser.restaurants

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Search for 'Italian' restaurants executed."""
        try:
            trajectory = self.get_trajectory()
            
            # Check if search was performed
            search_typed = False
            search_clicked = False
            
            for step in trajectory:
                action = step.get('action', {})
                action_type = action.get('action_type', '')
                params = action.get('parameters', {})
                
                if action_type == 'TYPE':
                    text = params.get('text', '').lower()
                    if 'italian' in text:
                        search_typed = True
                        
                elif action_type == 'CLICK':
                    selector = params.get('selector', '').lower()
                    if 'search' in selector:
                        search_clicked = True
            
            # Check final HTML for search results
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                if 'Search Results for "Italian"' in html_content or 'search Results for "Italian"' in html_content:
                    self.details['search_query'] = 'Italian'
                    return True
            
            if search_typed and search_clicked:
                self.issues.append("Search was executed but results page not verified")
                return True
                
            self.issues.append("Italian search was not properly executed")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Results sorted by delivery time."""
        try:
            trajectory = self.get_trajectory()
            
            # Check if sorting was performed
            sort_selected = False
            
            for step in trajectory:
                action = step.get('action', {})
                action_type = action.get('action_type', '')
                params = action.get('parameters', {})
                
                if action_type == 'SELECT':
                    label = params.get('label', '').lower()
                    if 'delivery time' in label:
                        sort_selected = True
                        self.details['sort_method'] = 'Delivery Time'
                        break
            
            if sort_selected:
                return True
            else:
                self.issues.append("Results were not sorted by delivery time")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking sort operation: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Fastest delivery restaurant identified."""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML not found")
                return False
            
            restaurants = self.parse_restaurants_from_html(final_html)
            
            if not restaurants:
                self.issues.append("No restaurants found in results")
                return False
            
            # Find restaurant with fastest badge or shortest delivery time
            fastest_restaurant = None
            
            # First, check for FASTEST badge
            for restaurant in restaurants:
                if restaurant.get('has_fastest_badge'):
                    fastest_restaurant = restaurant
                    break
            
            # If no badge, find by minimum delivery time
            if not fastest_restaurant:
                min_time = float('inf')
                for restaurant in restaurants:
                    time_str = restaurant.get('delivery_time', '')
                    match = re.search(r'(\d+)\s*min', time_str)
                    if match:
                        time_val = int(match.group(1))
                        if time_val < min_time:
                            min_time = time_val
                            fastest_restaurant = restaurant
            
            if fastest_restaurant:
                self.details['fastest_restaurant'] = fastest_restaurant['name']
                self.details['fastest_delivery_time'] = fastest_restaurant['delivery_time']
                return True
            else:
                self.issues.append("Could not identify fastest restaurant")
                return False
                
        except Exception as e:
            self.issues.append(f"Error identifying fastest restaurant: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Fastest restaurant added to favorites."""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML not found")
                return False
            
            restaurants = self.parse_restaurants_from_html(final_html)
            
            # Find the fastest restaurant
            fastest_restaurant = None
            for restaurant in restaurants:
                if restaurant.get('has_fastest_badge'):
                    fastest_restaurant = restaurant
                    break
            
            # If no badge, find by minimum delivery time
            if not fastest_restaurant:
                min_time = float('inf')
                for restaurant in restaurants:
                    time_str = restaurant.get('delivery_time', '')
                    match = re.search(r'(\d+)\s*min', time_str)
                    if match:
                        time_val = int(match.group(1))
                        if time_val < min_time:
                            min_time = time_val
                            fastest_restaurant = restaurant
            
            if not fastest_restaurant:
                self.issues.append("Could not identify fastest restaurant to verify favorite status")
                return False
            
            # Check if it's favorited
            if fastest_restaurant.get('is_favorited'):
                self.details['favorited_restaurant'] = fastest_restaurant['name']
                return True
            else:
                self.issues.append(f"Fastest restaurant '{fastest_restaurant['name']}' was not added to favorites")
                self.details['fastest_not_favorited'] = fastest_restaurant['name']
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking favorite status: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_italian_search_executed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_sorted_by_delivery_time'] = self.check_checkpoint_2()
            self.checkpoints['cp3_fastest_restaurant_identified'] = self.check_checkpoint_3()
            self.checkpoints['cp4_fastest_restaurant_favorited'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 2,
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
                'query_id': 2,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python food_query2.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

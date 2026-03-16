#!/usr/bin/env python3
# Evaluator for reservation Query 2

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class RestaurantDetailParser(HTMLParser):
    """Parse HTML to extract restaurant detail page information."""

    def __init__(self):
        super().__init__()
        self.in_title = False
        self.in_detail_title = False
        self.restaurant_name = None
        self.is_detail_page = False
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're on a detail page by looking for restaurant-detail-page class
        if tag == 'div' and 'class' in attrs_dict:
            if 'restaurant-detail-page' in attrs_dict['class']:
                self.is_detail_page = True
        
        # Check for detail title (h1 with class detail-title)
        if tag == 'h1' and 'class' in attrs_dict:
            if 'detail-title' in attrs_dict['class']:
                self.in_detail_title = True

    def handle_data(self, data):
        if self.in_detail_title:
            self.restaurant_name = data.strip()
            self.in_detail_title = False

    def handle_endtag(self, tag):
        pass


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints
        self.checkpoints = {
            "cp1_navigated_to_discover_page": False,
            "cp2_scrolled_to_find_last_restaurant": False,
            "cp3_identified_correct_last_restaurant": False,
            "cp4_viewed_restaurant_detail_page": False,
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
        """Checkpoint 1: Agent navigated to Discover Restaurants page (search page)"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                self.issues.append("No trajectory data found")
                return False

            # Check if agent visited the search/discover page
            for step in trajectory:
                page_info = step.get('page_info', {})
                url = page_info.get('url', '')
                
                # The discover restaurants page is the search page or home page
                if '/search' in url or url.endswith('/') or url.endswith('localhost:3005'):
                    self.details['discover_page_visited'] = True
                    return True
            
            self.issues.append("Agent did not navigate to Discover Restaurants page")
            return False

        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Agent scrolled down to find the last restaurant"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                return False

            # Count scroll actions and check if they scrolled significantly
            scroll_count = 0
            max_scroll_y = 0
            
            for step in trajectory:
                action = step.get('action', {})
                if action.get('action_type') == 'SCROLL':
                    scroll_count += 1
                
                page_info = step.get('page_info', {})
                viewport = page_info.get('viewport', {})
                scroll_y = viewport.get('scrollY', 0)
                max_scroll_y = max(max_scroll_y, scroll_y)
            
            # Agent should have scrolled multiple times and reached significant depth
            # Based on trajectory, the last restaurants appear around scrollY > 3000
            if scroll_count >= 5 and max_scroll_y > 3000:
                self.details['scroll_count'] = scroll_count
                self.details['max_scroll_position'] = max_scroll_y
                return True
            else:
                self.issues.append(f"Insufficient scrolling to reach last restaurant (scrolls: {scroll_count}, max Y: {max_scroll_y})")
                return False

        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Agent identified the correct last restaurant (Pacific Rim Bistro based on trajectory)"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                return False

            # Look through trajectory for clicks on restaurant detail pages
            # Based on the trajectory, the last restaurant should be Pacific Rim Bistro (ID 7889)
            for step in trajectory:
                page_info = step.get('page_info', {})
                url = page_info.get('url', '')
                
                # Check if they clicked on Pacific Rim Bistro's detail page
                # Based on the trajectory, restaurant ID 7889 is Pacific Rim Bistro
                if '/restaurant/7889' in url:
                    self.details['attempted_last_restaurant'] = 'Pacific Rim Bistro'
                    return True
            
            self.issues.append("Agent did not identify/click on the last restaurant")
            return False

        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Agent successfully viewed the restaurant detail page in final state"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = RestaurantDetailParser()
            parser.feed(html_content)

            # Check if we're on a detail page
            if not parser.is_detail_page:
                self.issues.append("Final page is not a restaurant detail page")
                return False

            # Check if we have a restaurant name
            if not parser.restaurant_name:
                self.issues.append("Could not find restaurant name on detail page")
                return False

            # Based on the final HTML showing Pacific Rim Bistro
            if 'Pacific Rim Bistro' in parser.restaurant_name:
                self.details['final_restaurant_name'] = parser.restaurant_name
                return True
            else:
                self.issues.append(f"Final page shows wrong restaurant: {parser.restaurant_name}")
                self.details['final_restaurant_name'] = parser.restaurant_name
                return False

        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_discover_page'] = self.check_checkpoint_1()
            self.checkpoints['cp2_scrolled_to_find_last_restaurant'] = self.check_checkpoint_2()
            self.checkpoints['cp3_identified_correct_last_restaurant'] = self.check_checkpoint_3()
            self.checkpoints['cp4_viewed_restaurant_detail_page'] = self.check_checkpoint_4()

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
        print("Usage: python reservation_query2.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

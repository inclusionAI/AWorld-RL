#!/usr/bin/env python3
# Evaluator for food Query 1

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class RestaurantHTMLParser(HTMLParser):
    """Parse HTML to extract restaurant information from search results."""

    def __init__(self):
        super().__init__()
        self.restaurants = []
        self.current_restaurant = {}
        self.in_result_card = False
        self.in_result_name = False
        self.in_rating_value = False
        self.in_favorite_btn = False
        self.current_favorite_classes = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Detect restaurant card
        if tag == 'div' and 'class' in attrs_dict and 'result-card' in attrs_dict['class']:
            self.in_result_card = True
            self.current_restaurant = {'name': '', 'rating': 0.0, 'is_favorited': False}
        
        # Detect restaurant name
        if self.in_result_card and tag == 'h3' and 'class' in attrs_dict and 'result-name' in attrs_dict['class']:
            self.in_result_name = True
        
        # Detect rating value
        if self.in_result_card and tag == 'span' and 'class' in attrs_dict and 'rating-value' in attrs_dict['class']:
            self.in_rating_value = True
        
        # Detect favorite button
        if self.in_result_card and tag == 'button' and 'class' in attrs_dict and 'favorite-btn' in attrs_dict['class']:
            self.in_favorite_btn = True
            self.current_favorite_classes = attrs_dict['class'].split()

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_result_name:
            self.current_restaurant['name'] = data
        
        if self.in_rating_value:
            try:
                self.current_restaurant['rating'] = float(data)
            except ValueError:
                pass

    def handle_endtag(self, tag):
        if tag == 'h3' and self.in_result_name:
            self.in_result_name = False
        
        if tag == 'span' and self.in_rating_value:
            self.in_rating_value = False
        
        if tag == 'button' and self.in_favorite_btn:
            self.in_favorite_btn = False
            # Check if favorited class is present
            self.current_restaurant['is_favorited'] = 'favorited' in self.current_favorite_classes
            self.current_favorite_classes = []
        
        if tag == 'div' and self.in_result_card:
            # End of restaurant card
            if self.current_restaurant.get('name') and 'rating' in self.current_restaurant:
                self.restaurants.append(self.current_restaurant)
            self.in_result_card = False
            self.current_restaurant = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        self.checkpoints = {
            "cp1_search_executed": False,
            "cp2_italian_restaurants_found": False,
            "cp3_all_high_rated_favorited": False,
            "cp4_no_low_rated_favorited": False,
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

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found in {self.result_dir}")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def parse_restaurants(self, html_path: Path) -> List[Dict]:
        """Parse restaurant data from HTML file."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = RestaurantHTMLParser()
        parser.feed(html_content)
        return parser.restaurants

    def check_search_query_in_url(self, html_content: str) -> bool:
        """Check if the HTML contains search results for 'Italian'."""
        # Check for search query in title or search input
        if re.search(r'Search Results for ["\']Italian["\']', html_content, re.IGNORECASE):
            return True
        if re.search(r'value=["\']Italian["\']', html_content, re.IGNORECASE):
            return True
        return False

    def check_checkpoint_1(self, html_path: Path) -> bool:
        """Checkpoint 1: Search was executed for 'Italian'"""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            has_search = self.check_search_query_in_url(html_content)
            
            if not has_search:
                self.issues.append("Search for 'Italian' was not executed or not visible in final state")
                return False
            
            return True
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def check_checkpoint_2(self, restaurants: List[Dict]) -> bool:
        """Checkpoint 2: Italian restaurants were found in results"""
        if not restaurants or len(restaurants) == 0:
            self.issues.append("No restaurants found in search results")
            return False
        
        self.details['total_restaurants_found'] = len(restaurants)
        return True

    def check_checkpoint_3(self, restaurants: List[Dict]) -> bool:
        """Checkpoint 3: All restaurants with rating >= 4.5 are favorited"""
        high_rated = [r for r in restaurants if r['rating'] >= 4.5]
        
        if not high_rated:
            self.issues.append("No restaurants with rating >= 4.5 found")
            return False
        
        self.details['high_rated_restaurants'] = [
            {'name': r['name'], 'rating': r['rating'], 'is_favorited': r['is_favorited']}
            for r in high_rated
        ]
        
        not_favorited = [r for r in high_rated if not r['is_favorited']]
        
        if not_favorited:
            self.issues.append(
                f"The following restaurants with rating >= 4.5 were NOT favorited: " +
                ", ".join([f"{r['name']} ({r['rating']}★)" for r in not_favorited])
            )
            return False
        
        return True

    def check_checkpoint_4(self, restaurants: List[Dict]) -> bool:
        """Checkpoint 4: No restaurants with rating < 4.5 are favorited"""
        low_rated = [r for r in restaurants if r['rating'] < 4.5]
        
        wrongly_favorited = [r for r in low_rated if r['is_favorited']]
        
        if wrongly_favorited:
            self.details['wrongly_favorited_restaurants'] = [
                {'name': r['name'], 'rating': r['rating']}
                for r in wrongly_favorited
            ]
            self.issues.append(
                f"The following restaurants with rating < 4.5 were incorrectly favorited: " +
                ", ".join([f"{r['name']} ({r['rating']}★)" for r in wrongly_favorited])
            )
            return False
        
        return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Find final HTML
            final_html = self.find_final_html()
            if not final_html:
                raise FileNotFoundError("No final HTML state file found")
            
            # Parse restaurants from final HTML
            restaurants = self.parse_restaurants(final_html)
            self.details['parsed_restaurants'] = [
                {'name': r['name'], 'rating': r['rating'], 'is_favorited': r['is_favorited']}
                for r in restaurants
            ]
            
            # Run checkpoint validations
            self.checkpoints['cp1_search_executed'] = self.check_checkpoint_1(final_html)
            self.checkpoints['cp2_italian_restaurants_found'] = self.check_checkpoint_2(restaurants)
            self.checkpoints['cp3_all_high_rated_favorited'] = self.check_checkpoint_3(restaurants)
            self.checkpoints['cp4_no_low_rated_favorited'] = self.check_checkpoint_4(restaurants)

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 1,
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
                'query_id': 1,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python food_query1.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

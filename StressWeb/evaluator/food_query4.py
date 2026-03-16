#!/usr/bin/env python3
# Evaluator for food Query 4

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class RestaurantHTMLParser(HTMLParser):
    """Parse HTML to extract restaurant and favorite information."""

    def __init__(self):
        super().__init__()
        self.in_restaurant = False
        self.in_restaurant_name = False
        self.in_cuisine = False
        self.in_price = False
        self.in_favorite_button = False
        self.current_restaurant = {}
        self.restaurants = []
        self.active_filters = []
        self.in_filter_tag = False
        self.results_count = None
        self.in_results_count = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for restaurant item
        if tag == 'div' and 'class' in attrs_dict and 'restaurant-item' in attrs_dict['class']:
            self.in_restaurant = True
            self.current_restaurant = {
                'name': '',
                'cuisine': '',
                'price': '',
                'is_favorited': False
            }
        
        # Check for restaurant name
        if self.in_restaurant and tag == 'h4' and 'class' in attrs_dict and 'restaurant-name' in attrs_dict['class']:
            self.in_restaurant_name = True
        
        # Check for cuisine tag
        if self.in_restaurant and tag == 'span' and 'class' in attrs_dict and attrs_dict['class'] == 'cuisine':
            self.in_cuisine = True
        
        # Check for price tag
        if self.in_restaurant and tag == 'span' and 'class' in attrs_dict and attrs_dict['class'] == 'price':
            self.in_price = True
        
        # Check for favorite button
        if self.in_restaurant and tag == 'button' and 'class' in attrs_dict and 'favorite-button' in attrs_dict['class']:
            self.in_favorite_button = True
            if 'favorited' in attrs_dict['class']:
                self.current_restaurant['is_favorited'] = True
        
        # Check for active filter tags
        if tag == 'span' and 'class' in attrs_dict and attrs_dict['class'] == 'filter-tag':
            self.in_filter_tag = True
        
        # Check for results count
        if tag == 'h3' and 'class' in attrs_dict and 'results-count' in attrs_dict['class']:
            self.in_results_count = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.in_restaurant_name:
            self.current_restaurant['name'] = data
        elif self.in_cuisine:
            self.current_restaurant['cuisine'] = data
        elif self.in_price:
            self.current_restaurant['price'] = data
        elif self.in_filter_tag:
            self.active_filters.append(data)
        elif self.in_results_count:
            self.results_count = data

    def handle_endtag(self, tag):
        if tag == 'h4':
            self.in_restaurant_name = False
        elif tag == 'span':
            self.in_cuisine = False
            self.in_price = False
            self.in_filter_tag = False
        elif tag == 'button':
            self.in_favorite_button = False
        elif tag == 'h3':
            self.in_results_count = False
        elif tag == 'div':
            if self.in_restaurant and self.current_restaurant.get('name'):
                self.restaurants.append(self.current_restaurant)
                self.current_restaurant = {}
                self.in_restaurant = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        self.checkpoints = {
            "cp1_mexican_cuisine_filter_applied": False,
            "cp2_price_range_filter_applied": False,
            "cp3_three_restaurants_favorited": False,
            "cp4_favorited_restaurants_match_criteria": False,
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
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def parse_html(self, html_file: Path) -> RestaurantHTMLParser:
        """Parse HTML file and return parser with extracted data."""
        parser = RestaurantHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self, parser: RestaurantHTMLParser) -> bool:
        """Checkpoint 1: Mexican cuisine filter applied"""
        if 'Mexican' in parser.active_filters:
            self.details['mexican_filter_applied'] = True
            return True
        else:
            self.issues.append("Mexican cuisine filter not applied")
            self.details['mexican_filter_applied'] = False
            return False

    def check_checkpoint_2(self, parser: RestaurantHTMLParser) -> bool:
        """Checkpoint 2: $$ price range filter applied"""
        if '$$' in parser.active_filters:
            self.details['price_filter_applied'] = True
            return True
        else:
            self.issues.append("$$ price range filter not applied")
            self.details['price_filter_applied'] = False
            return False

    def check_checkpoint_3(self, parser: RestaurantHTMLParser) -> bool:
        """Checkpoint 3: Exactly 3 restaurants saved to favorites"""
        favorited_restaurants = [r for r in parser.restaurants if r['is_favorited']]
        favorited_count = len(favorited_restaurants)
        
        self.details['favorited_count'] = favorited_count
        self.details['favorited_restaurants'] = [r['name'] for r in favorited_restaurants]
        
        if favorited_count == 3:
            return True
        elif favorited_count < 3:
            self.issues.append(f"Only {favorited_count} restaurants favorited, need 3")
            return False
        else:
            self.issues.append(f"Too many restaurants favorited: {favorited_count}, should be exactly 3")
            return False

    def check_checkpoint_4(self, parser: RestaurantHTMLParser) -> bool:
        """Checkpoint 4: Favorited restaurants match filter criteria (Mexican, $$)"""
        favorited_restaurants = [r for r in parser.restaurants if r['is_favorited']]
        
        if not favorited_restaurants:
            self.issues.append("No restaurants favorited to validate criteria")
            return False
        
        all_match = True
        mismatched = []
        
        for restaurant in favorited_restaurants:
            cuisine_match = restaurant.get('cuisine', '') == 'Mexican'
            price_match = restaurant.get('price', '') == '$$'
            
            if not (cuisine_match and price_match):
                all_match = False
                mismatched.append({
                    'name': restaurant['name'],
                    'cuisine': restaurant.get('cuisine', 'N/A'),
                    'price': restaurant.get('price', 'N/A')
                })
        
        self.details['all_favorited_match_criteria'] = all_match
        if mismatched:
            self.details['mismatched_restaurants'] = mismatched
            self.issues.append(f"Some favorited restaurants don't match criteria: {mismatched}")
        
        return all_match

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Find and parse final HTML
            final_html = self.find_final_html()
            if not final_html:
                raise FileNotFoundError("No final HTML file found")
            
            parser = self.parse_html(final_html)
            
            # Store parsed data in details
            self.details['total_restaurants_displayed'] = len(parser.restaurants)
            self.details['active_filters'] = parser.active_filters
            self.details['results_count_text'] = parser.results_count
            
            # Run checkpoint validations
            self.checkpoints['cp1_mexican_cuisine_filter_applied'] = self.check_checkpoint_1(parser)
            self.checkpoints['cp2_price_range_filter_applied'] = self.check_checkpoint_2(parser)
            self.checkpoints['cp3_three_restaurants_favorited'] = self.check_checkpoint_3(parser)
            self.checkpoints['cp4_favorited_restaurants_match_criteria'] = self.check_checkpoint_4(parser)
            
            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 4,
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
                'query_id': 4,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python food_query4.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Evaluator for reservation Query 1

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class RestaurantSearchParser(HTMLParser):
    """Parse HTML to extract restaurant information from search results."""

    def __init__(self):
        super().__init__()
        self.restaurants = []
        self.current_restaurant = {}
        self.in_restaurant_card = False
        self.in_name = False
        self.in_rating = False
        self.capture_text = False
        self.current_tag_stack = []

    def handle_starttag(self, tag, attrs):
        self.current_tag_stack.append(tag)
        attrs_dict = dict(attrs)
        
        # Detect restaurant cards
        if tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'restaurant-card' in classes or 'search-result-card' in classes:
                self.in_restaurant_card = True
                self.current_restaurant = {}
        
        # Capture restaurant name
        if self.in_restaurant_card and tag == 'h3':
            self.in_name = True
            self.capture_text = True
        
        # Capture rating
        if self.in_restaurant_card and tag == 'span' and 'class' in attrs_dict:
            if 'rating' in attrs_dict['class'].lower():
                self.in_rating = True
                self.capture_text = True

    def handle_data(self, data):
        if self.capture_text:
            data = data.strip()
            if data:
                if self.in_name:
                    self.current_restaurant['name'] = data
                elif self.in_rating:
                    # Extract numeric rating
                    match = re.search(r'(\d+\.?\d*)', data)
                    if match:
                        self.current_restaurant['rating'] = float(match.group(1))

    def handle_endtag(self, tag):
        if self.current_tag_stack:
            self.current_tag_stack.pop()
        
        if tag == 'h3' and self.in_name:
            self.in_name = False
            self.capture_text = False
        
        if tag == 'span' and self.in_rating:
            self.in_rating = False
            self.capture_text = False
        
        if tag == 'div' and self.in_restaurant_card:
            if self.current_restaurant:
                self.restaurants.append(self.current_restaurant)
            self.in_restaurant_card = False
            self.current_restaurant = {}


class RestaurantDetailParser(HTMLParser):
    """Parse HTML to extract restaurant detail page information."""

    def __init__(self):
        super().__init__()
        self.restaurant_name = None
        self.rating = None
        self.cuisine = None
        self.is_favorited = False
        self.in_title = False
        self.in_rating = False
        self.in_cuisine = False
        self.capture_text = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for favorited button
        if tag == 'button' and 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'favorite' in classes.lower() or 'favorited' in classes.lower():
                # Check button text will be captured in handle_data
                pass
        
        # Capture restaurant title
        if tag == 'h1' and 'class' in attrs_dict:
            if 'detail-title' in attrs_dict['class'] or 'restaurant-title' in attrs_dict['class']:
                self.in_title = True
                self.capture_text = True
        
        # Capture rating
        if tag == 'span' and 'class' in attrs_dict:
            if 'rating' in attrs_dict['class'].lower():
                self.in_rating = True
                self.capture_text = True
        
        # Capture cuisine
        if tag == 'span' and 'class' in attrs_dict:
            if 'meta-item' in attrs_dict['class']:
                self.in_cuisine = True
                self.capture_text = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        if self.capture_text:
            if self.in_title and not self.restaurant_name:
                self.restaurant_name = data
            elif self.in_rating and not self.rating:
                match = re.search(r'(\d+\.?\d*)', data)
                if match:
                    self.rating = float(match.group(1))
            elif self.in_cuisine and not self.cuisine:
                # Extract cuisine type (skip icons)
                if not data.startswith('fa-') and len(data) > 2 and data not in ['$$', '$$$', '$$$$']:
                    self.cuisine = data
        
        # Check for "Favorited" text
        if 'favorited' in data.lower():
            self.is_favorited = True

    def handle_endtag(self, tag):
        if tag == 'h1' and self.in_title:
            self.in_title = False
            self.capture_text = False
        if tag == 'span':
            if self.in_rating:
                self.in_rating = False
                self.capture_text = False
            if self.in_cuisine:
                self.in_cuisine = False
                self.capture_text = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        self.checkpoints = {
            "cp1_search_executed": False,
            "cp2_italian_cuisine_filtered": False,
            "cp3_highest_rated_identified": False,
            "cp4_restaurant_added_to_favorites": False,
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
        """Checkpoint 1: Search for 'Italian' restaurants was executed"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if search was performed with 'Italian'
            for action in action_history:
                if action.get('action', {}).get('action_type') == 'TYPE':
                    text = action.get('action', {}).get('parameters', {}).get('text', '')
                    if 'italian' in text.lower():
                        self.details['search_term'] = text
                        return True
            
            self.issues.append("No search for 'Italian' restaurants was found in action history")
            return False
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Search results show Italian cuisine restaurants"""
        try:
            # Check step 4 (after search is executed)
            step_files = self.find_step_html("step_4_*")
            if not step_files:
                step_files = self.find_step_html("step_3_*")
            
            if not step_files:
                self.issues.append("No search results page found")
                return False
            
            with open(step_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if Italian restaurants are present in results
            italian_keywords = ['bella italia', 'bella napoli', 'italian']
            found_italian = any(keyword in html_content.lower() for keyword in italian_keywords)
            
            if found_italian:
                self.details['search_results_page'] = str(step_files[0])
                return True
            else:
                self.issues.append("Search results do not show Italian restaurants")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking search results: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: The highest-rated Italian restaurant was identified"""
        try:
            # Parse search results to find highest rated restaurant
            step_files = self.find_step_html("step_4_*")
            if not step_files:
                step_files = self.find_step_html("step_3_*")
            
            if not step_files:
                self.issues.append("No search results page to verify highest rating")
                return False
            
            with open(step_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = RestaurantSearchParser()
            parser.feed(html_content)
            
            if not parser.restaurants:
                # Try to find rating information directly in HTML
                ratings = re.findall(r'⭐\s*(\d+\.?\d*)', html_content)
                if ratings:
                    ratings_float = [float(r) for r in ratings]
                    highest_rating = max(ratings_float)
                    self.details['highest_rating_found'] = highest_rating
                    
                    # Check if the selected restaurant has this rating
                    final_html = self.find_final_html()
                    if final_html:
                        with open(final_html, 'r', encoding='utf-8') as f:
                            final_content = f.read()
                        
                        # Look for the rating on the detail page
                        detail_ratings = re.findall(r'⭐\s*(\d+\.?\d*)', final_content)
                        if detail_ratings:
                            final_rating = float(detail_ratings[0])
                            self.details['selected_restaurant_rating'] = final_rating
                            
                            # Check if it matches the highest rating
                            if abs(final_rating - highest_rating) < 0.01:
                                return True
                            else:
                                self.issues.append(f"Selected restaurant rating ({final_rating}) is not the highest ({highest_rating})")
                                return False
                
                self.issues.append("Could not parse restaurant ratings from search results")
                return False
            
            # Find highest rated restaurant
            if parser.restaurants:
                highest_rated = max(parser.restaurants, key=lambda x: x.get('rating', 0))
                self.details['highest_rated_restaurant'] = highest_rated
                return True
            
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking highest-rated restaurant: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Restaurant was successfully added to favorites"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML state found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = RestaurantDetailParser()
            parser.feed(html_content)
            
            # Check if we're on a restaurant detail page
            if not parser.restaurant_name:
                self.issues.append("Not on a restaurant detail page in final state")
                return False
            
            self.details['final_restaurant'] = parser.restaurant_name
            
            # Check if the restaurant is favorited
            if parser.is_favorited:
                self.details['favorite_status'] = 'Favorited'
                return True
            else:
                # Also check for the favorited button in HTML directly
                if 'favorited' in html_content.lower():
                    self.details['favorite_status'] = 'Favorited (found in HTML)'
                    return True
                
                self.issues.append("Restaurant was not added to favorites (no 'Favorited' status found)")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking favorite status: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_search_executed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_italian_cuisine_filtered'] = self.check_checkpoint_2()
            self.checkpoints['cp3_highest_rated_identified'] = self.check_checkpoint_3()
            self.checkpoints['cp4_restaurant_added_to_favorites'] = self.check_checkpoint_4()

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
        print("Usage: python reservation_query1.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

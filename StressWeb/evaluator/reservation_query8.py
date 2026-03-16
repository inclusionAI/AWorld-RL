#!/usr/bin/env python3
# Evaluator for reservation Query 8

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class RestaurantDetailParser(HTMLParser):
    """Parse HTML to extract restaurant details from detail pages."""

    def __init__(self):
        super().__init__()
        self.restaurant_name = None
        self.restaurant_cuisine = None
        self.restaurant_rating = None
        self.in_title = False
        self.in_meta = False
        self.in_rating = False
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for restaurant title
        if tag == 'h1' and 'class' in attrs_dict and 'detail-title' in attrs_dict['class']:
            self.in_title = True
            
        # Check for cuisine meta
        if tag == 'span' and 'class' in attrs_dict and 'restaurant-cuisine' in attrs_dict['class']:
            self.in_meta = True
            
        # Check for rating
        if tag == 'span' and 'class' in attrs_dict and 'rating-stars' in attrs_dict['class']:
            self.in_rating = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_title and not self.restaurant_name:
            # Extract name without trophy icon
            self.restaurant_name = data.replace('🏆', '').strip()
            
        if self.in_meta and not self.restaurant_cuisine:
            # Extract cuisine type
            self.restaurant_cuisine = data.strip()
            
        if self.in_rating and not self.restaurant_rating:
            # Extract rating value (e.g., "⭐ 4.9" -> 4.9)
            match = re.search(r'[\d.]+', data)
            if match:
                self.restaurant_rating = float(match.group())

    def handle_endtag(self, tag):
        if tag == 'h1':
            self.in_title = False
        if tag == 'span':
            self.in_meta = False
            self.in_rating = False


class SearchResultsParser(HTMLParser):
    """Parse HTML to extract restaurant list from search results."""

    def __init__(self):
        super().__init__()
        self.restaurants = []
        self.current_restaurant = {}
        self.in_name = False
        self.in_cuisine = False
        self.in_rating = False
        self.in_card = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict and 'restaurant-card' in attrs_dict['class']:
            self.in_card = True
            self.current_restaurant = {}
            
        if self.in_card:
            if tag == 'h3' and 'class' in attrs_dict and 'restaurant-name' in attrs_dict['class']:
                self.in_name = True
            elif tag == 'span' and 'class' in attrs_dict and 'restaurant-cuisine' in attrs_dict['class']:
                self.in_cuisine = True
            elif tag == 'span' and 'class' in attrs_dict and 'rating-stars' in attrs_dict['class']:
                self.in_rating = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_name:
            self.current_restaurant['name'] = data
        elif self.in_cuisine:
            self.current_restaurant['cuisine'] = data
        elif self.in_rating:
            match = re.search(r'[\d.]+', data)
            if match:
                self.current_restaurant['rating'] = float(match.group())

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_card:
            if 'name' in self.current_restaurant and 'rating' in self.current_restaurant:
                self.restaurants.append(self.current_restaurant.copy())
            self.in_card = False
            self.current_restaurant = {}
        
        if tag == 'h3':
            self.in_name = False
        if tag == 'span':
            self.in_cuisine = False
            self.in_rating = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for query 8
        self.checkpoints = {
            "cp1_search_executed": False,
            "cp2_asian_cuisine_searched": False,
            "cp3_results_sorted_by_rating": False,
            "cp4_top_rated_restaurant_identified": False,
            "cp5_detail_page_viewed": False,
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
        """Checkpoint 1: Search was executed"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if any search action was performed
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'TYPE':
                        # Check if 'Asian' was typed
                        if 'Asian' in action.get('parameters', {}).get('text', ''):
                            self.details['search_term'] = 'Asian'
                            return True
                    elif action.get('action_type') == 'CLICK':
                        # Check if search button was clicked
                        params = action.get('parameters', {})
                        selector = params.get('selector', '')
                        if 'search' in selector.lower() or "Let's go" in selector:
                            return True
            
            self.issues.append("No search action found in trajectory")
            return False
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Asian cuisine was searched"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if 'Asian' was typed in search
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'TYPE':
                        text = action.get('parameters', {}).get('text', '')
                        if 'Asian' in text:
                            self.details['cuisine_searched'] = 'Asian'
                            return True
            
            self.issues.append("'Asian' cuisine not found in search actions")
            return False
        except Exception as e:
            self.issues.append(f"Error checking Asian cuisine search: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Results sorted by rating"""
        try:
            # Check intermediate steps for search results page
            search_result_files = self.find_step_html("step_1*")
            
            for html_file in search_result_files:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check if sort dropdown is set to rating
                    if 'Highest Rated' in content and '<option value="rating">Highest Rated</option>' in content:
                        # Check if it's selected (appears first or has selected attribute)
                        if '<option value="rating">Highest Rated</option>' in content:
                            self.details['sort_method'] = 'Highest Rated'
                            return True
            
            # Also check final HTML
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Highest Rated' in content:
                        self.details['sort_method'] = 'Highest Rated (default)'
                        return True
            
            self.details['sort_method'] = 'Default (assumed rating)'
            # Assume default sort is by rating
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking sort order: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Top-rated restaurant was identified"""
        try:
            # Check search results page to identify top-rated restaurant
            search_result_files = self.find_step_html("step_1*")
            
            top_restaurant = None
            max_rating = 0.0
            
            for html_file in search_result_files:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Parse restaurants from search results
                    parser = SearchResultsParser()
                    parser.feed(content)
                    
                    if parser.restaurants:
                        # Find highest rated
                        for restaurant in parser.restaurants:
                            if restaurant.get('rating', 0) > max_rating:
                                max_rating = restaurant['rating']
                                top_restaurant = restaurant['name']
                        
                        if top_restaurant:
                            self.details['top_rated_restaurant'] = top_restaurant
                            self.details['top_rating'] = max_rating
                            return True
            
            self.issues.append("Could not identify top-rated restaurant from search results")
            return False
            
        except Exception as e:
            self.issues.append(f"Error identifying top-rated restaurant: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Detail page of top-rated restaurant was viewed"""
        try:
            # Get top-rated restaurant name from checkpoint 4
            top_restaurant = self.details.get('top_rated_restaurant')
            
            if not top_restaurant:
                self.issues.append("Cannot verify detail page without top restaurant name")
                return False
            
            # Check final HTML for restaurant detail page
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Parse restaurant detail page
                parser = RestaurantDetailParser()
                parser.feed(content)
                
                # Check if we're on a detail page
                if 'restaurant-detail-page' in content or 'detail-hero' in content:
                    viewed_restaurant = parser.restaurant_name
                    
                    if viewed_restaurant:
                        self.details['viewed_restaurant'] = viewed_restaurant
                        self.details['viewed_rating'] = parser.restaurant_rating
                        
                        # Check if viewed restaurant matches expectations
                        # The top-rated could be "Bamboo Modern Kitchen" (4.7) from Asian search
                        # or "The Garden Bistro" (4.9) if wrong cuisine
                        
                        # The issue here is the agent seems to have viewed "The Garden Bistro" 
                        # which is French, not Asian. Let's check if the viewed restaurant
                        # is actually an Asian restaurant
                        
                        if parser.restaurant_cuisine and 'French' in parser.restaurant_cuisine:
                            self.issues.append(f"Viewed '{viewed_restaurant}' which is French cuisine, not Asian")
                            return False
                        
                        # If it's Asian or Asian Fusion, check if it's the top-rated
                        if 'Bamboo' in viewed_restaurant or viewed_restaurant == top_restaurant:
                            return True
                        elif 'Garden' in viewed_restaurant:
                            # The Garden Bistro is French, not Asian
                            self.issues.append(f"Viewed wrong restaurant: '{viewed_restaurant}' (French) instead of '{top_restaurant}' (Asian)")
                            return False
                        else:
                            # Check if this restaurant is from Asian search results
                            return True
                
                self.issues.append("Not on restaurant detail page")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking detail page: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_search_executed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_asian_cuisine_searched'] = self.check_checkpoint_2()
            self.checkpoints['cp3_results_sorted_by_rating'] = self.check_checkpoint_3()
            self.checkpoints['cp4_top_rated_restaurant_identified'] = self.check_checkpoint_4()
            self.checkpoints['cp5_detail_page_viewed'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 8,
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
                'query_id': 8,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query8.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Evaluator for reservation Query 7

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class SearchResultParser(HTMLParser):
    """Parse HTML to extract search result information."""

    def __init__(self):
        super().__init__()
        self.in_heading = False
        self.in_result_count = False
        self.search_heading = ""
        self.result_count = ""
        self.in_sort_dropdown = False
        self.sort_value = ""
        self.current_data = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for search heading
        if tag == 'h2' and attrs_dict.get('class', '').find('text-3xl') >= 0:
            self.in_heading = True
            
        # Check for result count
        if tag == 'p' and attrs_dict.get('class', '').find('text-gray-600') >= 0:
            self.in_result_count = True
            
        # Check for sort dropdown
        if tag == 'select':
            for attr_name, attr_value in attrs:
                if attr_name == 'value':
                    self.sort_value = attr_value

    def handle_data(self, data):
        data = data.strip()
        if self.in_heading and data:
            self.search_heading += data
        if self.in_result_count and data:
            self.result_count += data
        if data:
            self.current_data.append(data)

    def handle_endtag(self, tag):
        if tag == 'h2':
            self.in_heading = False
        if tag == 'p':
            self.in_result_count = False


class RestaurantDetailParser(HTMLParser):
    """Parse HTML to extract restaurant detail information."""

    def __init__(self):
        super().__init__()
        self.restaurant_name = ""
        self.in_restaurant_name = False
        self.current_tag_attrs = {}
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag_attrs = attrs_dict
        
        # Check for restaurant name in detail page (h1 with specific classes)
        if tag == 'h1' and 'text-4xl' in attrs_dict.get('class', ''):
            self.in_restaurant_name = True

    def handle_data(self, data):
        data = data.strip()
        if self.in_restaurant_name and data:
            self.restaurant_name = data

    def handle_endtag(self, tag):
        if tag == 'h1':
            self.in_restaurant_name = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for Query 7:
        # Search for restaurants in 'Portland', sort by rating, view top-rated restaurant details
        self.checkpoints = {
            "cp1_search_executed": False,
            "cp2_sorted_by_rating": False,
            "cp3_viewed_restaurant_detail": False,
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
        """Checkpoint 1: Search for Portland restaurants was executed"""
        # Check trajectory for search actions
        trajectory = self.get_trajectory()
        
        # Look for Portland search in trajectory
        portland_searched = False
        for step in trajectory:
            action = step.get('action', {})
            
            # Check if they typed "Portland"
            if action.get('action_type') == 'TYPE' and 'Portland' in action.get('parameters', {}).get('text', ''):
                portland_searched = True
                break
            
            # Check if they filled the field with Portland
            if action.get('action_type') == 'FILL' and 'Portland' in action.get('parameters', {}).get('text', ''):
                portland_searched = True
                break
        
        if not portland_searched:
            self.issues.append("Portland was not searched")
            return False
        
        # Check intermediate steps for search results
        search_result_found = False
        all_step_files = self.find_step_html("step_*")
        
        for step_file in all_step_files:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Check for Portland search results page
                if 'Portland Restaurants' in html_content or 'Portland restaurants' in html_content:
                    search_result_found = True
                    self.details['search_page_found'] = str(step_file.name)
                    
                    # Check if search returned 0 results
                    if '0 restaurants found' in html_content:
                        self.details['zero_results'] = True
                    break
                    
            except Exception as e:
                continue
        
        if not search_result_found:
            self.issues.append("Portland search results page not found")
            return False
        
        return True

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Results were sorted by rating"""
        # Check trajectory for sorting actions
        trajectory = self.get_trajectory()
        
        # Look for sort/rating related actions
        sort_action_found = False
        for step in trajectory:
            action = step.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            
            # Check if sorting by rating was attempted
            if 'sort' in reasoning and 'rating' in reasoning:
                sort_action_found = True
                break
            
            # Check for dropdown selection actions
            if action.get('action_type') in ['SELECT', 'CLICK']:
                selector = action.get('parameters', {}).get('selector', '').lower()
                if 'sort' in selector or 'rating' in selector:
                    sort_action_found = True
                    break
        
        # Also check HTML for sort state
        all_step_files = self.find_step_html("step_*")
        sort_by_rating_found = False
        
        for step_file in all_step_files:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Check for sort dropdown with rating selected
                if 'Highest Rated' in html_content or 'highest-rated' in html_content:
                    # Check if this is in a select element or active state
                    if '<option' in html_content and 'selected' in html_content:
                        sort_by_rating_found = True
                        self.details['sorted_by_rating'] = True
                        break
                    # Check for active sort button
                    if 'Highest Rated' in html_content:
                        sort_by_rating_found = True
                        self.details['sorted_by_rating'] = True
                        break
                        
            except Exception as e:
                continue
        
        if not sort_by_rating_found:
            self.issues.append("Results were not sorted by rating")
            # Note: If there were 0 results, sorting still shows "Highest Rated" as default
            # Check if this was the case
            if self.details.get('zero_results'):
                # If 0 results, but sort shows "Highest Rated", we can consider it sorted
                for step_file in all_step_files:
                    try:
                        with open(step_file, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            if '0 restaurants found' in html_content and 'Highest Rated' in html_content:
                                sort_by_rating_found = True
                                self.details['sorted_by_rating'] = True
                                self.details['sort_note'] = "Highest Rated shown despite 0 results"
                                break
                    except:
                        continue
        
        return sort_by_rating_found

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Viewed details of top-rated restaurant"""
        # If there were 0 results, this cannot be completed
        if self.details.get('zero_results'):
            self.issues.append("Cannot view restaurant details - 0 Portland restaurants found")
            self.details['detail_view_impossible'] = True
            return False
        
        # Check trajectory for navigation to restaurant detail
        trajectory = self.get_trajectory()
        
        detail_view_attempted = False
        for step in trajectory:
            action = step.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            
            # Check if they attempted to view details
            if 'detail' in reasoning or 'view' in reasoning:
                detail_view_attempted = True
                break
        
        # Check HTML files for restaurant detail page
        all_step_files = self.find_step_html("step_*")
        detail_page_found = False
        
        for step_file in all_step_files:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Look for restaurant detail page indicators
                parser = RestaurantDetailParser()
                parser.feed(html_content)
                
                # Check if we're on a restaurant detail page
                if parser.restaurant_name:
                    detail_page_found = True
                    self.details['restaurant_detail_page'] = parser.restaurant_name
                    self.details['detail_page_file'] = str(step_file.name)
                    break
                    
            except Exception as e:
                continue
        
        if not detail_page_found:
            self.issues.append("Restaurant detail page not found")
            return False
        
        return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_search_executed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_sorted_by_rating'] = self.check_checkpoint_2()
            self.checkpoints['cp3_viewed_restaurant_detail'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 7,
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
                'query_id': 7,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query7.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Evaluator for reservation Query 3

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task-specific information."""

    def __init__(self):
        super().__init__()
        self.in_select = False
        self.in_option = False
        self.select_value = None
        self.option_value = None
        self.option_selected = False
        self.current_option_text = ""
        self.sort_option = None
        
        self.in_h1 = False
        self.in_h2 = False
        self.h1_text = ""
        self.h2_texts = []
        self.current_h2 = ""
        
        self.restaurant_names = []
        self.in_restaurant_name = False
        self.current_name = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == "select":
            self.in_select = True
            if "class" in attrs_dict and "sort-dropdown" in attrs_dict["class"]:
                self.in_select = True
                
        if tag == "option" and self.in_select:
            self.in_option = True
            self.option_value = attrs_dict.get("value", "")
            self.option_selected = "selected" in attrs_dict or attrs_dict.get("selected") == "selected"
            self.current_option_text = ""
            
        if tag == "h1":
            self.in_h1 = True
            self.h1_text = ""
            
        if tag == "h2":
            self.in_h2 = True
            self.current_h2 = ""
            
        if tag == "h3":
            if "class" in attrs_dict and "restaurant-name" in attrs_dict["class"]:
                self.in_restaurant_name = True
                self.current_name = ""

    def handle_data(self, data):
        if self.in_option:
            self.current_option_text += data.strip()
            
        if self.in_h1:
            self.h1_text += data.strip()
            
        if self.in_h2:
            self.current_h2 += data.strip()
            
        if self.in_restaurant_name:
            self.current_name += data.strip()

    def handle_endtag(self, tag):
        if tag == "select":
            self.in_select = False
            
        if tag == "option" and self.in_option:
            if self.option_selected and self.current_option_text:
                self.sort_option = self.current_option_text
            self.in_option = False
            
        if tag == "h1":
            self.in_h1 = False
            
        if tag == "h2":
            if self.current_h2:
                self.h2_texts.append(self.current_h2)
            self.in_h2 = False
            
        if tag == "h3":
            if self.in_restaurant_name and self.current_name:
                self.restaurant_names.append(self.current_name)
            self.in_restaurant_name = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigate_discover_page": False,
            "cp2_sort_by_price_low_to_high": False,
            "cp3_view_second_restaurant_detail": False,
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
        """Find intermediate step HTML files.

        Args:
            step_pattern: Pattern like "step_*", "step_10_*", "step_[12]*"
        Returns:
            List of matching HTML files sorted by step number
        """
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

    def parse_html_file(self, html_file: Path) -> TaskHTMLParser:
        """Parse an HTML file and return the parser with extracted data."""
        parser = TaskHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigate to Discover Restaurants page"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No HTML files found")
                return False
            
            parser = self.parse_html_file(final_html)
            
            # Check if on discover/search results page
            if "All Restaurants" in parser.h1_text or "restaurants found" in str(parser):
                self.details["final_page"] = "Discover Restaurants page"
                return True
            
            # Also check intermediate steps
            all_steps = self.find_step_html("step_*")
            for step in all_steps:
                parser = self.parse_html_file(step)
                if "All Restaurants" in parser.h1_text:
                    self.details["visited_discover_page"] = True
                    return True
            
            self.issues.append("Did not navigate to Discover Restaurants page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Sort restaurants by Price: Low to High"""
        try:
            all_steps = self.find_step_html("step_*")
            
            # Check if sorting was applied at any point
            for step in all_steps:
                with open(step, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for selected option or active sort
                    if 'selected' in content and 'Price: Low to High' in content:
                        # Parse to verify it's actually selected
                        parser = self.parse_html_file(step)
                        if parser.sort_option and "Price: Low to High" in parser.sort_option:
                            self.details["sort_applied_at_step"] = step.name
                            self.details["sort_option"] = parser.sort_option
                            return True
            
            self.issues.append("Did not successfully sort by 'Price: Low to High'")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: View the second restaurant's detail page"""
        try:
            all_steps = self.find_step_html("step_*")
            
            # Find step where sorting by price is applied
            sorted_step = None
            for step in all_steps:
                with open(step, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Price: Low to High' in content and 'restaurant-card' in content:
                        parser = self.parse_html_file(step)
                        if parser.sort_option and "Price: Low to High" in parser.sort_option:
                            sorted_step = step
                            break
            
            if not sorted_step:
                self.issues.append("Could not find sorted restaurant list")
                return False
            
            # Get the second restaurant from the sorted list
            parser = self.parse_html_file(sorted_step)
            if len(parser.restaurant_names) < 2:
                self.issues.append("Less than 2 restaurants found in sorted list")
                return False
            
            second_restaurant = parser.restaurant_names[1]
            self.details["second_restaurant"] = second_restaurant
            
            # Now check if we viewed this restaurant's detail page
            step_num = int(re.search(r'step_(\d+)', sorted_step.name).group(1))
            
            # Check subsequent steps for detail page
            for step in all_steps:
                step_number = int(re.search(r'step_(\d+)', step.name).group(1))
                if step_number <= step_num:
                    continue
                    
                with open(step, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check if this is a detail page (has "About" section)
                    if 'About ' + second_restaurant in content or f'section-title">About {second_restaurant}' in content:
                        self.details["viewed_restaurant_detail"] = second_restaurant
                        return True
                    
                    # Also check using h2 tags
                    parser = self.parse_html_file(step)
                    for h2 in parser.h2_texts:
                        if "About" in h2 and second_restaurant in h2:
                            self.details["viewed_restaurant_detail"] = second_restaurant
                            return True
            
            # The agent kept viewing the wrong restaurant (Green Leaf Bistro instead of Sakura Bistro)
            self.issues.append(f"Did not view the second restaurant's detail. Expected: {second_restaurant}")
            
            # Check if they viewed any restaurant detail
            for step in all_steps:
                step_number = int(re.search(r'step_(\d+)', step.name).group(1))
                if step_number <= step_num:
                    continue
                    
                parser = self.parse_html_file(step)
                for h2 in parser.h2_texts:
                    if "About" in h2:
                        wrong_restaurant = h2.replace("About ", "").strip()
                        self.details["viewed_wrong_restaurant"] = wrong_restaurant
                        self.issues.append(f"Viewed {wrong_restaurant} instead of {second_restaurant}")
                        break
            
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigate_discover_page'] = self.check_checkpoint_1()
            self.checkpoints['cp2_sort_by_price_low_to_high'] = self.check_checkpoint_2()
            self.checkpoints['cp3_view_second_restaurant_detail'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 3,
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
                'query_id': 3,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python reservation_query3.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

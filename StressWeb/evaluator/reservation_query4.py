#!/usr/bin/env python3
# Evaluator for reservation Query 4

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
        self.in_page_class = False
        self.page_class = None
        self.in_menu_section = False
        self.in_section_title = False
        self.in_menu_category = False
        self.in_menu_item = False
        self.menu_categories = []
        self.menu_items = []
        self.current_tag = None
        self.current_attrs = {}
        self.section_title_text = ""
        self.found_menu_page = False
        self.found_menu_categories = False
        
        # Restaurant detail page tracking
        self.on_detail_page = False
        self.restaurant_name = ""
        self.in_restaurant_title = False
        self.in_cta_section = False
        self.has_view_menu_button = False

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        # Check for restaurant detail page
        if tag == 'div':
            div_class = self.current_attrs.get('class', '')
            if 'restaurant-detail-page' in div_class:
                self.on_detail_page = True
            elif 'menu-page' in div_class:
                self.found_menu_page = True
            elif 'detail-title' in div_class:
                self.in_restaurant_title = True
            elif 'detail-cta' in div_class or 'cta-buttons' in div_class:
                self.in_cta_section = True
            elif 'menu-category' in div_class:
                self.in_menu_category = True
            elif 'menu-item' in div_class:
                self.in_menu_item = True
        
        # Check for menu button
        if tag == 'button' and self.in_cta_section:
            btn_class = self.current_attrs.get('class', '')
            if 'menu-btn' in btn_class:
                self.has_view_menu_button = True
        
        # Track section titles
        if tag == 'h2':
            h2_class = self.current_attrs.get('class', '')
            if 'section-title' in h2_class:
                self.in_section_title = True
                self.section_title_text = ""
        
        # Track menu categories
        if tag == 'h3' and self.in_menu_category:
            pass

    def handle_endtag(self, tag):
        if tag == 'h2' and self.in_section_title:
            self.in_section_title = False
            if self.section_title_text.strip():
                if 'menu' in self.section_title_text.lower():
                    self.in_menu_section = True
        
        if tag == 'div':
            if self.in_restaurant_title:
                self.in_restaurant_title = False
            if self.in_cta_section:
                self.in_cta_section = False
            if self.in_menu_category:
                self.in_menu_category = False

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        
        # Capture restaurant name
        if self.in_restaurant_title and self.current_tag == 'h1':
            self.restaurant_name = data
        
        # Capture section titles
        if self.in_section_title:
            self.section_title_text += data
        
        # Track menu categories
        if self.in_menu_category and self.current_tag == 'h3':
            self.menu_categories.append(data)
            self.found_menu_categories = True
        
        # Track menu items
        if self.in_menu_item:
            self.menu_items.append(data)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task: "View the menu of The Garden Bistro restaurant"
        self.checkpoints = {
            "cp1_navigated_to_garden_bistro": False,
            "cp2_found_view_menu_button": False,
            "cp3_clicked_view_menu": False,
            "cp4_reached_menu_page": False,
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

    def parse_html_file(self, html_file: Path) -> TaskHTMLParser:
        """Parse an HTML file and return the parser with extracted data."""
        parser = TaskHTMLParser()
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                parser.feed(content)
        except Exception as e:
            self.issues.append(f"Error parsing {html_file.name}: {str(e)}")
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to The Garden Bistro detail page"""
        try:
            # Check if any step has The Garden Bistro detail page
            all_steps = sorted(self.result_dir.glob("step_*_raw.html"))
            
            for step_file in all_steps:
                parser = self.parse_html_file(step_file)
                if parser.on_detail_page and 'Garden Bistro' in parser.restaurant_name:
                    self.details['garden_bistro_page_found'] = step_file.name
                    return True
            
            self.issues.append("Never navigated to The Garden Bistro detail page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Found the View Menu button on restaurant detail page"""
        try:
            # Check if The Garden Bistro page has a View Menu button
            all_steps = sorted(self.result_dir.glob("step_*_raw.html"))
            
            for step_file in all_steps:
                parser = self.parse_html_file(step_file)
                if parser.on_detail_page and 'Garden Bistro' in parser.restaurant_name:
                    if parser.has_view_menu_button:
                        self.details['view_menu_button_found'] = step_file.name
                        return True
            
            self.issues.append("View Menu button not found on The Garden Bistro detail page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Clicked the View Menu button"""
        try:
            # Check trajectory for menu button click
            trajectory = self.get_trajectory()
            
            for entry in trajectory:
                action = entry.get('action', {})
                action_type = action.get('action_type', '')
                parameters = action.get('parameters', {})
                
                if action_type == 'CLICK':
                    selector = parameters.get('selector', '')
                    # Check if clicked menu button
                    if 'menu-btn' in selector.lower() or 'menu' in selector.lower():
                        result = entry.get('result', {})
                        if result.get('success'):
                            self.details['menu_button_clicked'] = True
                            return True
            
            self.issues.append("Agent never clicked the View Menu button")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Successfully reached menu page with menu content"""
        try:
            # Check final HTML and recent steps for menu page
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False
            
            # Check last few steps for menu page
            all_steps = sorted(self.result_dir.glob("step_*_raw.html"))
            steps_to_check = all_steps[-10:] if len(all_steps) > 10 else all_steps
            steps_to_check.append(final_html)
            
            for step_file in steps_to_check:
                parser = self.parse_html_file(step_file)
                
                # Check if reached menu page
                if parser.found_menu_page:
                    self.details['menu_page_reached'] = step_file.name
                    
                    # Verify menu has actual content (categories or items)
                    if parser.found_menu_categories or len(parser.menu_items) > 0:
                        self.details['menu_categories'] = parser.menu_categories
                        self.details['menu_items_count'] = len(parser.menu_items)
                        return True
                    else:
                        self.issues.append("Menu page reached but no menu content found")
                        return False
            
            # Check final state
            final_parser = self.parse_html_file(final_html)
            if final_parser.on_detail_page:
                self.issues.append("Agent remained on restaurant detail page, never reached menu page")
            else:
                self.issues.append("Agent did not reach the menu page")
            
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_garden_bistro'] = self.check_checkpoint_1()
            self.checkpoints['cp2_found_view_menu_button'] = self.check_checkpoint_2()
            self.checkpoints['cp3_clicked_view_menu'] = self.check_checkpoint_3()
            self.checkpoints['cp4_reached_menu_page'] = self.check_checkpoint_4()

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
        print("Usage: python reservation_query4.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

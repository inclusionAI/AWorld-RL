#!/usr/bin/env python3
# Evaluator for file Query 10

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class FileDetailsHTMLParser(HTMLParser):
    """Parse HTML to extract file details information."""

    def __init__(self):
        super().__init__()
        self.in_search_results = False
        self.in_file_details = False
        self.search_query = None
        self.file_found_in_search = False
        self.current_file_name = None
        self.file_details_visible = False
        self.file_info_data = {}
        self.current_class = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get('class', '')
        
        # Check if we're in search results
        if 'search-results-container' in class_name:
            self.in_search_results = True
        
        # Check if file details panel is visible
        if 'file-details-main' in class_name or 'file-details-content' in class_name:
            self.in_file_details = True
            self.file_details_visible = True
        
        # Check if we're in a result item
        if 'result-item' in class_name or 'file-name' in class_name:
            self.current_class = class_name

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        
        # Check search query
        if 'welcome_kickoff.pptx' in text.lower():
            if self.in_search_results and not self.in_file_details:
                self.file_found_in_search = True
                if 'welcome_kickoff.pptx' == text:
                    self.current_file_name = text
            elif self.in_file_details:
                self.current_file_name = text


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_search_executed": False,
            "cp2_file_found": False,
            "cp3_details_button_clicked": False,
            "cp4_file_details_visible": False,
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

    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data from traj.jsonl."""
        if not self.traj_file.exists():
            return []

        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                trajectory.append(json.loads(line))
        return trajectory

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Search was executed for welcome_kickoff.pptx"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if search was performed
            search_performed = False
            search_text = None
            
            for action in action_history:
                action_data = action.get('action', {})
                action_type = action_data.get('action_type', '')
                
                # Look for FILL action with search input
                if action_type == 'FILL':
                    params = action_data.get('parameters', {})
                    selector = params.get('selector', '')
                    text = params.get('text', '')
                    
                    if 'search' in selector.lower() and 'welcome_kickoff.pptx' in text.lower():
                        search_text = text
                        search_performed = True
                        break
            
            if search_performed:
                self.details['search_text'] = search_text
                return True
            else:
                self.issues.append("Search for 'welcome_kickoff.pptx' was not executed")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: File was found in search results"""
        try:
            html_file = self.find_final_html()
            if not html_file:
                self.issues.append("No HTML file found to verify search results")
                return False
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML to check if file is found
            parser = FileDetailsHTMLParser()
            parser.feed(html_content)
            
            if parser.file_found_in_search:
                self.details['file_found'] = True
                self.details['file_name'] = parser.current_file_name or 'welcome_kickoff.pptx'
                return True
            else:
                self.issues.append("File 'welcome_kickoff.pptx' not found in search results")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking search results: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: View details button was clicked"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Look for click actions on the view/details button
            details_button_clicked = False
            click_count = 0
            
            for action in action_history:
                action_data = action.get('action', {})
                action_type = action_data.get('action_type', '')
                
                if action_type == 'CLICK':
                    params = action_data.get('parameters', {})
                    reasoning = action_data.get('reasoning', '')
                    
                    # Check if this is a click on the details/visibility button
                    if any(keyword in reasoning.lower() for keyword in ['view', 'detail', 'visibility', 'eye']):
                        details_button_clicked = True
                        click_count += 1
            
            if details_button_clicked:
                self.details['details_button_clicks'] = click_count
                return True
            else:
                self.issues.append("View details button was not clicked")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking button clicks: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: File details panel is visible"""
        try:
            html_file = self.find_final_html()
            if not html_file:
                self.issues.append("No HTML file found to verify file details")
                return False
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML to check if file details are visible
            parser = FileDetailsHTMLParser()
            parser.feed(html_content)
            
            # Check for file details panel
            if parser.file_details_visible:
                self.details['details_panel_visible'] = True
                return True
            else:
                # Also check raw HTML for file details classes
                if 'file-details-main' in html_content or 'file-details-content' in html_content:
                    # Check if it's actually displayed (not hidden)
                    if 'display: none' not in html_content or 'file-details' in html_content:
                        self.details['details_panel_visible'] = True
                        return True
                
                self.issues.append("File details panel is not visible in the final state")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking file details visibility: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_search_executed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_file_found'] = self.check_checkpoint_2()
            self.checkpoints['cp3_details_button_clicked'] = self.check_checkpoint_3()
            self.checkpoints['cp4_file_details_visible'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 10,
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
                'query_id': 10,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python file_query10.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Evaluator for note Query 11

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class SearchResultsParser(HTMLParser):
    """Parse HTML to extract search results information."""

    def __init__(self):
        super().__init__()
        self.in_results_meta = False
        self.in_strong = False
        self.current_data = []
        self.query_text = None
        self.filters_text = None
        self.results_count = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'p' and attrs_dict.get('class') == 'results-meta':
            self.in_results_meta = True
        elif tag == 'strong' and self.in_results_meta:
            self.in_strong = True

    def handle_endtag(self, tag):
        if tag == 'p' and self.in_results_meta:
            self.in_results_meta = False
        elif tag == 'strong':
            if self.in_strong and self.current_data:
                data = ''.join(self.current_data).strip()
                if self.query_text is None:
                    self.query_text = data
                elif self.filters_text is None:
                    self.filters_text = data
                self.current_data = []
            self.in_strong = False

    def handle_data(self, data):
        if self.in_strong:
            self.current_data.append(data)


class AdvancedSearchFormParser(HTMLParser):
    """Parse Advanced Search form to check selected filters."""

    def __init__(self):
        super().__init__()
        self.search_query = None
        self.date_range_selected = None
        self.selected_tags = []
        self.in_tag_chip = False
        self.current_tag_text = []
        self.current_tag_active = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check search input
        if tag == 'input' and attrs_dict.get('placeholder') == 'Enter keywords to search...':
            self.search_query = attrs_dict.get('value', '').strip()
        
        # Check radio buttons for date range
        if tag == 'input' and attrs_dict.get('type') == 'radio' and attrs_dict.get('name') == 'dateRange':
            if 'checked' in attrs_dict or attrs_dict.get('checked') == '':
                self.date_range_selected = attrs_dict.get('value')
        
        # Check tag chips
        if tag == 'button' and 'tag-chip' in attrs_dict.get('class', ''):
            self.in_tag_chip = True
            self.current_tag_text = []
            # Check if the tag chip has the 'active' class
            if 'active' in attrs_dict.get('class', ''):
                self.current_tag_active = True
            else:
                self.current_tag_active = False

    def handle_endtag(self, tag):
        if tag == 'button' and self.in_tag_chip:
            if self.current_tag_active and self.current_tag_text:
                # Extract tag name (remove icon and count)
                tag_text = ''.join(self.current_tag_text).strip()
                # Remove tag count like "(1)"
                tag_text = re.sub(r'\s*\(\d+\)\s*$', '', tag_text)
                self.selected_tags.append(tag_text)
            self.in_tag_chip = False
            self.current_tag_text = []
            self.current_tag_active = False

    def handle_data(self, data):
        if self.in_tag_chip:
            self.current_tag_text.append(data)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_advanced_search": False,
            "cp2_search_query_set_to_project": False,
            "cp3_date_range_all_time_selected": False,
            "cp4_tags_ai_and_academic_selected": False,
            "cp5_search_executed_and_results_displayed": False,
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
        """Checkpoint 1: Navigated to Advanced Search page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if any step navigated to advanced search
            for entry in trajectory:
                if 'url' in entry:
                    url = entry['url'].lower()
                    if 'search/advanced' in url or 'advanced' in url:
                        self.details['advanced_search_url_found'] = entry['url']
                        return True
            
            # Also check action history for clicking "Advanced Search"
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            for action in action_history:
                if action.get('action', {}).get('parameters', {}).get('selector', '').lower() == 'text=advanced search':
                    return True
            
            self.issues.append("Did not navigate to Advanced Search page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Search query set to 'project'"""
        try:
            # Check in intermediate steps (before search is executed)
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Look for search input with value "project"
                if 'placeholder="Enter keywords to search..."' in html_content:
                    # Extract value attribute
                    match = re.search(r'placeholder="Enter keywords to search\.\.\."[^>]*value="([^"]*)"', html_content)
                    if match:
                        value = match.group(1)
                        if value.lower() == 'project':
                            self.details['search_query'] = value
                            return True
            
            # Also check final HTML
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    # Check in results page meta
                    if '"project"' in html_content:
                        self.details['search_query'] = 'project'
                        return True
            
            self.issues.append("Search query was not set to 'project'")
            return False
        except Exception as e:
            self.issues.append(f"Error checking search query: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Date range 'All time' selected"""
        try:
            # Check in intermediate steps (step 2 after clicking "All time")
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                    # Parse the form to find selected date range
                    parser = AdvancedSearchFormParser()
                    parser.feed(html_content)
                    
                    if parser.date_range_selected == 'all_time':
                        self.details['date_range_selected'] = 'all_time'
                        return True
            
            # The final result page shows "Last 30 days" which means the selection didn't persist
            # Check the trajectory to see if "All time" was clicked
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            all_time_clicked = False
            for action in action_history:
                selector = action.get('action', {}).get('parameters', {}).get('selector', '')
                if 'all time' in selector.lower() or selector == 'text=All time':
                    all_time_clicked = True
                    break
            
            if not all_time_clicked:
                self.issues.append("'All time' date range was not clicked")
                return False
            
            # Check final results page to see if "All time" is in the filter display
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    if 'All time' in html_content or 'all time' in html_content.lower():
                        self.details['date_range_in_results'] = 'All time'
                        return True
            
            self.issues.append("Date range 'All time' was clicked but not reflected in final results (showing 'Last 30 days')")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking date range: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Tags 'ai' and 'academic' selected"""
        try:
            # Check final results page for selected tags
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML not found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Look for tags in the results meta information
            # The format should be like: "ai, academic tags" or similar
            if 'ai, academic' in html_content:
                self.details['selected_tags'] = ['ai', 'academic']
                return True
            
            # Check if both tags appear in the filter display
            has_ai = 'ai' in html_content.lower()
            has_academic = 'academic' in html_content.lower()
            
            if has_ai and has_academic:
                # Verify they were selected in the action history
                result_data = self.load_result_json()
                action_history = result_data.get('action_history', [])
                
                ai_clicked = False
                academic_clicked = False
                
                for action in action_history:
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '').lower()
                    if 'ai' in selector and action.get('action', {}).get('action_type') == 'CLICK':
                        ai_clicked = True
                    if 'academic' in selector and action.get('action', {}).get('action_type') == 'CLICK':
                        academic_clicked = True
                
                if ai_clicked and academic_clicked:
                    self.details['selected_tags'] = ['ai', 'academic']
                    return True
                else:
                    missing = []
                    if not ai_clicked:
                        missing.append('ai')
                    if not academic_clicked:
                        missing.append('academic')
                    self.issues.append(f"Tags not clicked: {', '.join(missing)}")
                    return False
            
            self.issues.append("Tags 'ai' and 'academic' were not both selected")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking tags: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Search executed and results displayed"""
        try:
            # Check if search button was clicked
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            search_clicked = False
            for action in action_history:
                if action.get('action', {}).get('action_type') == 'CLICK':
                    selector = action.get('action', {}).get('parameters', {}).get('selector', '').lower()
                    if 'search' in selector:
                        search_clicked = True
                        break
            
            if not search_clicked:
                self.issues.append("Search button was not clicked")
                return False
            
            # Check final HTML for search results page
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML not found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Look for search results indicators
            if 'Search Results' in html_content or 'search-results' in html_content:
                # Check for results count or empty results message
                if 'results found' in html_content or 'No results found' in html_content or 'documents found' in html_content:
                    self.details['results_page_displayed'] = True
                    
                    # Extract results count if available
                    match = re.search(r'(\d+)\s+(?:results|documents)\s+found', html_content)
                    if match:
                        self.details['results_count'] = match.group(1)
                    
                    return True
            
            self.issues.append("Search results page was not displayed")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_advanced_search'] = self.check_checkpoint_1()
            self.checkpoints['cp2_search_query_set_to_project'] = self.check_checkpoint_2()
            self.checkpoints['cp3_date_range_all_time_selected'] = self.check_checkpoint_3()
            self.checkpoints['cp4_tags_ai_and_academic_selected'] = self.check_checkpoint_4()
            self.checkpoints['cp5_search_executed_and_results_displayed'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 11,
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
                'query_id': 11,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python note_query11.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

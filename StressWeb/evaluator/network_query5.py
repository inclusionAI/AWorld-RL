#!/usr/bin/env python3
# Evaluator for network Query 5

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class JobDetailParser(HTMLParser):
    """Parse HTML to extract job detail page information."""

    def __init__(self):
        super().__init__()
        self.in_job_detail = False
        self.in_h2 = False
        self.job_title = None
        self.current_tag = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check if we're in a job detail page
        if tag == 'div' and attrs_dict.get('class') == 'job-detail-page':
            self.in_job_detail = True
        
        if self.in_job_detail:
            if tag == 'h2':
                self.in_h2 = True
                self.current_tag = 'h2'
    
    def handle_data(self, data):
        if self.in_job_detail and self.in_h2:
            text = data.strip()
            if text and not self.job_title:
                self.job_title = text
    
    def handle_endtag(self, tag):
        if tag == 'h2':
            self.in_h2 = False


class JobSearchParser(HTMLParser):
    """Parse HTML to extract job search results."""

    def __init__(self):
        super().__init__()
        self.in_search_input = False
        self.search_value = None
        self.job_titles = []
        self.in_job_item = False
        self.in_job_h3 = False
        self.current_job_title = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for search input field with value
        if tag == 'input' and 'Search title, company, location' in attrs_dict.get('placeholder', ''):
            self.search_value = attrs_dict.get('value', '')
        
        # Check for job item
        if tag == 'div' and 'job-item' in attrs_dict.get('class', ''):
            self.in_job_item = True
            self.current_job_title = None
        
        # Check for job title h3
        if self.in_job_item and tag == 'h3':
            self.in_job_h3 = True
    
    def handle_data(self, data):
        if self.in_job_h3:
            text = data.strip()
            if text:
                self.current_job_title = text
    
    def handle_endtag(self, tag):
        if tag == 'h3':
            self.in_job_h3 = False
        if tag == 'div' and self.in_job_item:
            if self.current_job_title:
                self.job_titles.append(self.current_job_title)
            self.in_job_item = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_jobs": False,
            "cp2_searched_for_product": False,
            "cp3_search_returned_results": False,
            "cp4_clicked_first_result": False,
            "cp5_viewing_job_details": False,
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
        """Checkpoint 1: Navigated to Jobs page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check trajectory for navigation to /jobs
            for entry in trajectory:
                if 'page_info' in entry:
                    url = entry['page_info'].get('url', '')
                    if '/jobs' in url:
                        self.details['jobs_page_visited'] = True
                        return True
            
            self.issues.append("Did not navigate to Jobs page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Jobs navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Searched for 'Product'"""
        try:
            # Check step 2 (after filling search input) for search value
            step_files = self.find_step_html("step_2_*")
            if not step_files:
                self.issues.append("Step 2 HTML not found")
                return False
            
            with open(step_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = JobSearchParser()
            parser.feed(html_content)
            
            if parser.search_value and 'Product' in parser.search_value:
                self.details['search_term'] = parser.search_value
                return True
            
            self.issues.append(f"Search input did not contain 'Product', found: {parser.search_value}")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking search input: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Search returned results with Product jobs"""
        try:
            # Check step 3 (after clicking search button) for search results
            step_files = self.find_step_html("step_3_*")
            if not step_files:
                self.issues.append("Step 3 HTML not found")
                return False
            
            with open(step_files[0], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = JobSearchParser()
            parser.feed(html_content)
            
            # Check if any job titles contain "Product"
            product_jobs = [title for title in parser.job_titles if 'Product' in title]
            
            if product_jobs:
                self.details['search_results'] = parser.job_titles
                self.details['product_jobs_found'] = product_jobs
                return True
            
            self.issues.append(f"No Product-related jobs in search results. Found: {parser.job_titles}")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking search results: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Clicked the first search result"""
        try:
            trajectory = self.get_trajectory()
            
            # Look for click action on job title in trajectory
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'CLICK':
                        selector = action.get('parameters', {}).get('selector', '')
                        # Check if clicked on a job title (h3 or job link)
                        if 'h3' in selector or 'job-link' in selector:
                            # Verify navigation to job detail page
                            if 'page_info' in entry:
                                url = entry['page_info'].get('url', '')
                                if '/jobs/' in url and url != 'http://localhost:3010/jobs':
                                    self.details['job_detail_url'] = url
                                    return True
            
            self.issues.append("Did not click on a job result to view details")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking job click: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Viewing job details page with Product job"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML not found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = JobDetailParser()
            parser.feed(html_content)
            
            # Check if we're on a job detail page
            if not parser.in_job_detail:
                self.issues.append("Not on a job detail page")
                return False
            
            # Check if the job title contains "Product"
            if parser.job_title and 'Product' in parser.job_title:
                self.details['final_job_title'] = parser.job_title
                return True
            
            self.issues.append(f"Job detail page does not show a Product-related job. Found: {parser.job_title}")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking job details: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_jobs'] = self.check_checkpoint_1()
            self.checkpoints['cp2_searched_for_product'] = self.check_checkpoint_2()
            self.checkpoints['cp3_search_returned_results'] = self.check_checkpoint_3()
            self.checkpoints['cp4_clicked_first_result'] = self.check_checkpoint_4()
            self.checkpoints['cp5_viewing_job_details'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 5,
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
                'query_id': 5,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python network_query5.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

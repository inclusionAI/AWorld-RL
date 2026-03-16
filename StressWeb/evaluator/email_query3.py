#!/usr/bin/env python3
# Evaluator for email Query 3

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class EmailSearchParser(HTMLParser):
    """Parse HTML to extract email search results information."""

    def __init__(self):
        super().__init__()
        self.in_search_header = False
        self.in_search_info = False
        self.in_filter_chip = False
        self.in_results_count = False
        self.in_message_item = False
        self.in_sender = False
        
        self.search_query = None
        self.active_filter = None
        self.results_count = None
        self.email_senders = []
        self.current_sender = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and attrs_dict.get('class') == 'search-header':
            self.in_search_header = True
        elif tag == 'div' and attrs_dict.get('class') == 'search-info':
            self.in_search_info = True
        elif tag == 'button' and 'filter-chip' in attrs_dict.get('class', ''):
            self.in_filter_chip = True
            if 'active' in attrs_dict.get('class', ''):
                self.current_filter_active = True
            else:
                self.current_filter_active = False
        elif tag == 'p' and attrs_dict.get('class') == 'results-count':
            self.in_results_count = True
        elif tag == 'div' and attrs_dict.get('class') and 'message-item' in attrs_dict.get('class', ''):
            self.in_message_item = True
        elif tag == 'span' and attrs_dict.get('class') and 'sender' in attrs_dict.get('class', ''):
            self.in_sender = True

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_search_info and 'Search results for' in data:
            match = re.search(r'Search results for "([^"]+)"', data)
            if match:
                self.search_query = match.group(1)
        elif self.in_filter_chip and data and self.current_filter_active:
            self.active_filter = data
        elif self.in_results_count:
            match = re.search(r'(\d+)\s+emails?\s+found', data)
            if match:
                self.results_count = int(match.group(1))
        elif self.in_sender and data:
            self.current_sender = data

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_search_header:
                self.in_search_header = False
            if self.in_search_info:
                self.in_search_info = False
            if self.in_message_item:
                if self.current_sender:
                    self.email_senders.append(self.current_sender)
                    self.current_sender = None
                self.in_message_item = False
        elif tag == 'button' and self.in_filter_chip:
            self.in_filter_chip = False
            self.current_filter_active = False
        elif tag == 'p' and self.in_results_count:
            self.in_results_count = False
        elif tag == 'span' and self.in_sender:
            self.in_sender = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_search_executed": False,
            "cp2_correct_sender_filter": False,
            "cp3_date_filter_applied": False,
            "cp4_emails_deleted": False,
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

    def parse_html_search_state(self, html_file: Path) -> Dict:
        """Parse HTML to extract search state."""
        if not html_file.exists():
            return {}
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = EmailSearchParser()
        parser.feed(html_content)
        
        return {
            'search_query': parser.search_query,
            'active_filter': parser.active_filter,
            'results_count': parser.results_count,
            'email_senders': parser.email_senders
        }

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Search was executed for hr@company.com"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML state found")
            return False
        
        search_state = self.parse_html_search_state(final_html)
        
        if not search_state.get('search_query'):
            self.issues.append("No search query was executed")
            return False
        
        if search_state['search_query'] != 'from:hr@company.com':
            self.issues.append(f"Wrong search query: {search_state['search_query']}, expected 'from:hr@company.com'")
            return False
        
        self.details['search_query'] = search_state['search_query']
        return True

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Correct sender filter applied (hr@company.com)"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        search_state = self.parse_html_search_state(final_html)
        
        if search_state.get('results_count') is None:
            self.issues.append("No search results found")
            return False
        
        # Check that all displayed emails are from hr@company.com
        email_senders = search_state.get('email_senders', [])
        if not email_senders:
            self.issues.append("No email senders found in search results")
            return False
        
        non_hr_senders = [s for s in email_senders if s != 'hr@company.com']
        if non_hr_senders:
            self.issues.append(f"Search results contain emails from wrong senders: {non_hr_senders}")
            return False
        
        self.details['email_senders'] = email_senders
        return True

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Date filter applied to 'Last 7 days'"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        search_state = self.parse_html_search_state(final_html)
        
        if search_state.get('active_filter') != 'Last 7 days':
            self.issues.append(f"Wrong date filter: {search_state.get('active_filter')}, expected 'Last 7 days'")
            return False
        
        self.details['date_filter'] = search_state['active_filter']
        return True

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: All emails from hr@company.com in last 7 days were deleted"""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        search_state = self.parse_html_search_state(final_html)
        
        # Check if search is still active and showing results
        if search_state.get('search_query') != 'from:hr@company.com':
            self.issues.append("Search context lost - cannot verify deletion")
            return False
        
        if search_state.get('active_filter') != 'Last 7 days':
            self.issues.append("Date filter changed - cannot verify deletion")
            return False
        
        # If emails are successfully deleted, results_count should be 0
        results_count = search_state.get('results_count')
        if results_count is None:
            self.issues.append("Cannot determine number of remaining emails")
            return False
        
        if results_count > 0:
            self.issues.append(f"{results_count} emails still present - deletion failed")
            self.details['remaining_emails'] = results_count
            return False
        
        self.details['remaining_emails'] = 0
        return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_search_executed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_correct_sender_filter'] = self.check_checkpoint_2()
            self.checkpoints['cp3_date_filter_applied'] = self.check_checkpoint_3()
            self.checkpoints['cp4_emails_deleted'] = self.check_checkpoint_4()

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
        print("Usage: python email_query3.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

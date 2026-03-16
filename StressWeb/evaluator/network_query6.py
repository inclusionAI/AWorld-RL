#!/usr/bin/env python3
# Evaluator for network Query 6

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract connection request information."""

    def __init__(self):
        super().__init__()
        self.in_invite_card = False
        self.in_invite_info = False
        self.in_h3 = False
        self.in_empty_state = False
        self.current_name = None
        self.connection_requests = []
        self.has_no_pending_message = False
        self.request_count = None
        self.connections_count = None
        self.in_stat_label = False
        self.in_stat_value = False
        self.current_stat_label = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and attrs_dict.get('class') == 'invite-card':
            self.in_invite_card = True
            self.current_name = None
        elif tag == 'div' and attrs_dict.get('class') == 'invite-info':
            self.in_invite_info = True
        elif tag == 'h3' and self.in_invite_info:
            self.in_h3 = True
        elif tag == 'div' and attrs_dict.get('class') == 'empty-state':
            self.in_empty_state = True
        elif tag == 'span' and attrs_dict.get('class') == 'stat-label':
            self.in_stat_label = True
        elif tag == 'span' and attrs_dict.get('class') == 'stat-value':
            self.in_stat_value = True

    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_invite_card:
                self.in_invite_card = False
                if self.current_name:
                    self.connection_requests.append(self.current_name)
            if self.in_invite_info:
                self.in_invite_info = False
            if self.in_empty_state:
                self.in_empty_state = False
        elif tag == 'h3':
            self.in_h3 = False
        elif tag == 'span':
            if self.in_stat_label:
                self.in_stat_label = False
            if self.in_stat_value:
                self.in_stat_value = False
                self.current_stat_label = None

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_h3 and self.in_invite_info:
            self.current_name = data
        elif self.in_empty_state and 'No pending' in data:
            self.has_no_pending_message = True
        elif self.in_stat_label:
            self.current_stat_label = data
        elif self.in_stat_value and self.current_stat_label:
            if self.current_stat_label == 'Requests':
                try:
                    self.request_count = int(data)
                except ValueError:
                    pass
            elif self.current_stat_label == 'Connections':
                try:
                    self.connections_count = int(data)
                except ValueError:
                    pass


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_network": False,
            "cp2_john_anderson_accepted": False,
            "cp3_other_requests_rejected": False,
            "cp4_no_pending_requests": False,
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

    def parse_html_for_requests(self, html_path: Path) -> Tuple[List[str], bool, int, int]:
        """Parse HTML to extract connection request information."""
        parser = TaskHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        
        return (
            parser.connection_requests,
            parser.has_no_pending_message,
            parser.request_count or 0,
            parser.connections_count or 0
        )

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to Network page"""
        trajectory = self.get_trajectory()
        
        for entry in trajectory:
            page_info = entry.get('page_info', {})
            url = page_info.get('url', '')
            if '/network' in url:
                self.details['navigated_to_network'] = True
                return True
        
        self.issues.append("Did not navigate to Network page")
        self.details['navigated_to_network'] = False
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: John Anderson's connection request was accepted"""
        # Check intermediate steps to see if John Anderson was present and then disappeared
        step_files = self.find_step_html("step_*")
        
        john_was_present = False
        john_disappeared = False
        
        for step_file in step_files:
            requests, _, _, _ = self.parse_html_for_requests(step_file)
            
            if 'John Anderson' in requests:
                john_was_present = True
            elif john_was_present and 'John Anderson' not in requests:
                john_disappeared = True
                break
        
        if john_was_present and john_disappeared:
            self.details['john_anderson_accepted'] = True
            return True
        
        if not john_was_present:
            self.issues.append("John Anderson was not found in connection requests")
        elif not john_disappeared:
            self.issues.append("John Anderson's request was not accepted (still present in requests)")
        
        self.details['john_anderson_accepted'] = False
        return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: All other requests were rejected"""
        # Check that Priya Patel and Emma Wilson are not present in final state
        step_files = self.find_step_html("step_*")
        
        # Get initial requests
        if step_files:
            initial_requests, _, _, _ = self.parse_html_for_requests(step_files[0])
            self.details['initial_requests'] = initial_requests
        else:
            self.issues.append("No step files found to verify initial state")
            return False
        
        # Get final requests
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML found")
            return False
        
        final_requests, _, _, _ = self.parse_html_for_requests(final_html)
        self.details['final_requests'] = final_requests
        
        # Check that other people (not John Anderson) were rejected
        other_people = [name for name in initial_requests if name != 'John Anderson']
        
        if not other_people:
            # If there were no other people, this checkpoint is satisfied
            self.details['other_requests_rejected'] = True
            return True
        
        # Check if any other people are still in final requests
        remaining_others = [name for name in other_people if name in final_requests]
        
        if remaining_others:
            self.issues.append(f"Other requests not rejected: {', '.join(remaining_others)}")
            self.details['other_requests_rejected'] = False
            return False
        
        self.details['other_requests_rejected'] = True
        return True

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: No pending connection requests remain"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML found")
            return False
        
        requests, has_no_pending, request_count, connections_count = self.parse_html_for_requests(final_html)
        
        self.details['final_request_count'] = request_count
        self.details['final_connections_count'] = connections_count
        self.details['has_no_pending_message'] = has_no_pending
        
        if has_no_pending and request_count == 0:
            self.details['no_pending_requests'] = True
            return True
        
        if not has_no_pending:
            self.issues.append("No 'No pending connection requests' message found")
        if request_count != 0:
            self.issues.append(f"Request count is {request_count}, expected 0")
        
        self.details['no_pending_requests'] = False
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_network'] = self.check_checkpoint_1()
            self.checkpoints['cp2_john_anderson_accepted'] = self.check_checkpoint_2()
            self.checkpoints['cp3_other_requests_rejected'] = self.check_checkpoint_3()
            self.checkpoints['cp4_no_pending_requests'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 6,
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
                'query_id': 6,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python network_query6.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

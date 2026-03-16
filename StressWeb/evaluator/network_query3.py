#!/usr/bin/env python3
# Evaluator for network Query 3

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class NetworkHTMLParser(HTMLParser):
    """Parse HTML to extract network page information."""

    def __init__(self):
        super().__init__()
        self.in_stat_label = False
        self.in_stat_value = False
        self.last_stat_label = None
        self.requests_count = None
        self.in_empty_state = False
        self.empty_state_text = None
        self.accept_button_count = 0
        self.in_tab_badge = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'span' and attrs_dict.get('class') == 'stat-label':
            self.in_stat_label = True
        elif tag == 'span' and attrs_dict.get('class') == 'stat-value':
            self.in_stat_value = True
        elif tag == 'div' and attrs_dict.get('class') == 'empty-state':
            self.in_empty_state = True
        elif tag == 'button' and 'accept-button' in attrs_dict.get('class', ''):
            self.accept_button_count += 1
        elif tag == 'span' and attrs_dict.get('class') == 'tab-badge':
            self.in_tab_badge = True

    def handle_endtag(self, tag):
        if tag == 'span':
            self.in_stat_label = False
            self.in_stat_value = False
            self.in_tab_badge = False
        elif tag == 'div':
            self.in_empty_state = False

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_stat_label:
            self.last_stat_label = data
        elif self.in_stat_value and self.last_stat_label == 'Requests':
            try:
                self.requests_count = int(data)
            except ValueError:
                pass
            self.last_stat_label = None
        elif self.in_empty_state:
            if 'No pending connection requests' in data:
                self.empty_state_text = data


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigated_to_network_page": False,
            "cp2_accepted_all_requests": False,
            "cp3_requests_count_zero": False,
            "cp4_empty_state_message_shown": False,
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
            step_pattern: Pattern like "step_*", "step_1_*", "step_[12]*"
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
                if line.strip():
                    trajectory.append(json.loads(line))
        return trajectory

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def parse_html(self, html_path: Path) -> NetworkHTMLParser:
        """Parse HTML file and return parser with extracted data."""
        parser = NetworkHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to Network page"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                self.issues.append("No trajectory data found")
                return False

            # Check if agent navigated to network page
            for entry in trajectory:
                if 'page_info' in entry:
                    url = entry['page_info'].get('url', '')
                    if '/network' in url:
                        self.details['navigated_to_network'] = True
                        return True

            self.issues.append("Agent did not navigate to Network page")
            return False

        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Accepted all pending requests (3 accept actions)"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                self.issues.append("No trajectory data for checking accept actions")
                return False

            # Count accept button clicks
            accept_clicks = 0
            for entry in trajectory:
                if 'action' in entry:
                    action = entry['action']
                    if action.get('action_type') == 'CLICK':
                        selector = action.get('parameters', {}).get('selector', '')
                        if 'Accept' in selector or 'accept' in selector.lower():
                            accept_clicks += 1

            self.details['accept_clicks'] = accept_clicks

            # Check initial state to see how many requests there were
            step_1_files = self.find_step_html("step_1_*")
            if step_1_files:
                parser = self.parse_html(step_1_files[0])
                initial_requests = parser.requests_count
                self.details['initial_requests_count'] = initial_requests
                
                if initial_requests and accept_clicks >= initial_requests:
                    return True
                else:
                    self.issues.append(f"Only {accept_clicks} accept actions performed, expected at least {initial_requests}")
                    return False
            else:
                # Fallback: check if at least 3 accept actions (based on task description)
                if accept_clicks >= 3:
                    return True
                else:
                    self.issues.append(f"Only {accept_clicks} accept actions performed, expected at least 3")
                    return False

        except Exception as e:
            self.issues.append(f"Error checking accept actions: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Requests count is 0 in final state"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            parser = self.parse_html(final_html)
            
            if parser.requests_count is not None:
                self.details['final_requests_count'] = parser.requests_count
                
                if parser.requests_count == 0:
                    return True
                else:
                    self.issues.append(f"Final requests count is {parser.requests_count}, expected 0")
                    return False
            else:
                self.issues.append("Could not find requests count in final HTML")
                return False

        except Exception as e:
            self.issues.append(f"Error checking requests count: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: 'No pending connection requests' message is shown"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found for empty state check")
                return False

            parser = self.parse_html(final_html)
            
            if parser.empty_state_text:
                self.details['empty_state_message'] = parser.empty_state_text
                return True
            
            # Alternative: check raw HTML content
            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'No pending connection requests' in content:
                    self.details['empty_state_message'] = 'Found in HTML'
                    return True

            self.issues.append("'No pending connection requests' message not found in final state")
            return False

        except Exception as e:
            self.issues.append(f"Error checking empty state message: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_network_page'] = self.check_checkpoint_1()
            self.checkpoints['cp2_accepted_all_requests'] = self.check_checkpoint_2()
            self.checkpoints['cp3_requests_count_zero'] = self.check_checkpoint_3()
            self.checkpoints['cp4_empty_state_message_shown'] = self.check_checkpoint_4()

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
        print("Usage: python network_query3.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

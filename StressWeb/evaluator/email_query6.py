#!/usr/bin/env python3
# Evaluator for email Query 6

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class PersonalLabelParser(HTMLParser):
    """Parse HTML to extract Personal label information."""

    def __init__(self):
        super().__init__()
        self.in_personal_section = False
        self.in_message_count = False
        self.personal_email_count = None
        self.current_tag = None
        self.current_attrs = []
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        # Check if we're in the Personal label section
        if tag == "span" and self.current_attrs.get("class") == "message-count":
            self.in_message_count = True

    def handle_data(self, data):
        # Extract count from message-count span like "(1)" or "(0)"
        if self.in_message_count:
            match = re.search(r'\((\d+)\)', data.strip())
            if match:
                self.personal_email_count = int(match.group(1))
                
    def handle_endtag(self, tag):
        if tag == "span":
            self.in_message_count = False


class TrashCountParser(HTMLParser):
    """Parse HTML to extract Trash folder count."""

    def __init__(self):
        super().__init__()
        self.trash_count = None
        self.in_trash_link = False
        self.in_count_span = False
        self.depth = 0
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Look for Trash nav link
        if tag == "a" and "nav-link" in attrs_dict.get("class", ""):
            self.depth += 1
            
        # Check if this is inside trash link and is a count span
        if self.depth > 0 and tag == "span" and "count" in attrs_dict.get("class", ""):
            self.in_count_span = True

    def handle_data(self, data):
        # Check if we're looking at Trash text
        if self.depth > 0 and "Trash" in data:
            self.in_trash_link = True
            
        # If we found Trash and now we're in a count span, capture the count
        if self.in_trash_link and self.in_count_span:
            try:
                self.trash_count = int(data.strip())
                self.in_trash_link = False  # Reset after capturing
            except ValueError:
                pass
                
    def handle_endtag(self, tag):
        if tag == "span":
            self.in_count_span = False
        if tag == "a":
            if self.depth > 0:
                self.depth -= 1
                if self.depth == 0:
                    self.in_trash_link = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigated_to_personal_label": False,
            "cp2_personal_label_view_displayed": False,
            "cp3_all_personal_emails_deleted": False,
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

    def parse_personal_count(self, html_content: str) -> Optional[int]:
        """Parse HTML to extract Personal label email count."""
        parser = PersonalLabelParser()
        try:
            parser.feed(html_content)
            return parser.personal_email_count
        except Exception as e:
            self.issues.append(f"Error parsing Personal label count: {str(e)}")
            return None

    def parse_trash_count(self, html_content: str) -> Optional[int]:
        """Parse HTML to extract Trash folder count."""
        # Use regex approach for more reliable extraction
        match = re.search(r'<span>Trash</span><span class="count[^"]*">(\d+)</span>', html_content)
        if match:
            return int(match.group(1))
        return None

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Agent navigated to Personal label"""
        trajectory = self.get_trajectory()
        
        # Look for clicks on Personal label in trajectory
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'CLICK':
                selector = action.get('parameters', {}).get('selector', '')
                reasoning = action.get('reasoning', '')
                
                # Check if clicking on Personal label
                if 'Personal' in selector or 'Personal' in reasoning:
                    self.details['personal_click_found'] = True
                    return True
        
        self.issues.append("Agent did not navigate to Personal label")
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Personal label view was displayed at some point"""
        # Check intermediate steps for Personal label view
        step_files = self.find_step_html("step_*")
        
        for step_file in step_files:
            try:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                # Look for Personal label view indicators
                if 'class="label-view-main"' in html_content and 'class="label-title">Personal</h1>' in html_content:
                    count = self.parse_personal_count(html_content)
                    self.details['personal_label_view_found'] = True
                    self.details['personal_count_in_view'] = count
                    return True
                    
            except Exception as e:
                continue
        
        self.issues.append("Personal label view was never displayed")
        return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: All Personal emails were deleted (count = 0)"""
        final_html = self.find_final_html()
        
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
        
        try:
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if we're in Personal label view in final state
            in_personal_view = 'class="label-view-main"' in html_content and 'class="label-title">Personal</h1>' in html_content
            
            if in_personal_view:
                # Parse the count from Personal label view
                count = self.parse_personal_count(html_content)
                self.details['final_personal_count'] = count
                
                if count == 0:
                    self.details['all_personal_emails_deleted'] = True
                    return True
                else:
                    self.issues.append(f"Personal label still has {count} email(s)")
                    return False
            else:
                # Not in Personal view, check sidebar count or try to find it
                # Look for Personal label in sidebar
                match = re.search(r'<span>Personal</span>.*?<span class="count[^"]*">(\d+)</span>', html_content, re.DOTALL)
                if match:
                    # Personal label doesn't show count in sidebar by default, so we need to check the label view
                    # Since we're not in the label view, we need to check if any intermediate step shows 0
                    step_files = self.find_step_html("step_*")
                    
                    for step_file in reversed(step_files):
                        try:
                            with open(step_file, 'r', encoding='utf-8') as f:
                                step_html = f.read()
                            
                            if 'class="label-view-main"' in step_html and 'class="label-title">Personal</h1>' in step_html:
                                count = self.parse_personal_count(step_html)
                                self.details['final_personal_count'] = count
                                
                                if count == 0:
                                    self.details['all_personal_emails_deleted'] = True
                                    return True
                                else:
                                    self.issues.append(f"Personal label still has {count} email(s)")
                                    return False
                        except Exception:
                            continue
                
                self.issues.append("Could not verify Personal label count in final state")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking final state: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_personal_label'] = self.check_checkpoint_1()
            self.checkpoints['cp2_personal_label_view_displayed'] = self.check_checkpoint_2()
            self.checkpoints['cp3_all_personal_emails_deleted'] = self.check_checkpoint_3()

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
        print("Usage: python email_query6.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

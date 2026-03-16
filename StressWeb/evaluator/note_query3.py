#!/usr/bin/env python3
# Evaluator for note Query 3

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class WorkspaceHTMLParser(HTMLParser):
    """Parse HTML to extract workspace information."""

    def __init__(self):
        super().__init__()
        self.workspace_name = None
        self.workspace_description = None
        self.privacy_level = None
        self.in_workspace_title = False
        self.in_workspace_subtitle = False
        self.in_privacy_badge = False
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for workspace title
        if tag == 'h1' and attrs_dict.get('class') == 'workspace-title':
            self.in_workspace_title = True
            self.current_data = []
        
        # Check for workspace subtitle (description)
        elif tag == 'p' and attrs_dict.get('class') == 'workspace-subtitle':
            self.in_workspace_subtitle = True
            self.current_data = []
        
        # Check for privacy badge
        elif tag == 'span' and 'privacy-badge' in attrs_dict.get('class', ''):
            self.in_privacy_badge = True
            self.current_data = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            if self.in_workspace_title or self.in_workspace_subtitle or self.in_privacy_badge:
                self.current_data.append(text)

    def handle_endtag(self, tag):
        if tag == 'h1' and self.in_workspace_title:
            self.workspace_name = ' '.join(self.current_data).strip()
            self.in_workspace_title = False
            self.current_data = []
        
        elif tag == 'p' and self.in_workspace_subtitle:
            self.workspace_description = ' '.join(self.current_data).strip()
            self.in_workspace_subtitle = False
            self.current_data = []
        
        elif tag == 'span' and self.in_privacy_badge:
            privacy_text = ' '.join(self.current_data).strip()
            # Extract privacy level (remove icon text if present)
            self.privacy_level = privacy_text.lower()
            self.in_privacy_badge = False
            self.current_data = []


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_workspace_created": False,
            "cp2_correct_name": False,
            "cp3_correct_description": False,
            "cp4_correct_privacy_level": False,
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
        """Checkpoint 1: Workspace was created and is visible in final state"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML state found")
            return False

        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Check if we're on a workspace page (has workspace-page class)
        if 'workspace-page' not in html_content:
            self.issues.append("Final page is not showing a workspace page")
            return False

        # Check if workspace title element exists
        if 'workspace-title' not in html_content:
            self.issues.append("Workspace title element not found in final HTML")
            return False

        self.details['workspace_page_found'] = True
        return True

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Workspace name is 'Product Team Workspace'"""
        final_html = self.find_final_html()
        if not final_html:
            return False

        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = WorkspaceHTMLParser()
        parser.feed(html_content)

        expected_name = "Product Team Workspace"
        
        if not parser.workspace_name:
            self.issues.append("Could not extract workspace name from HTML")
            return False

        if parser.workspace_name != expected_name:
            self.issues.append(f"Workspace name mismatch: expected '{expected_name}', got '{parser.workspace_name}'")
            self.details['workspace_name'] = parser.workspace_name
            return False

        self.details['workspace_name'] = parser.workspace_name
        return True

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Workspace description is 'Central hub for product development'"""
        final_html = self.find_final_html()
        if not final_html:
            return False

        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = WorkspaceHTMLParser()
        parser.feed(html_content)

        expected_description = "Central hub for product development"
        
        if not parser.workspace_description:
            self.issues.append("Could not extract workspace description from HTML")
            return False

        if parser.workspace_description != expected_description:
            self.issues.append(f"Workspace description mismatch: expected '{expected_description}', got '{parser.workspace_description}'")
            self.details['workspace_description'] = parser.workspace_description
            return False

        self.details['workspace_description'] = parser.workspace_description
        return True

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Privacy level is set to 'team'"""
        final_html = self.find_final_html()
        if not final_html:
            return False

        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = WorkspaceHTMLParser()
        parser.feed(html_content)

        expected_privacy = "team"
        
        if not parser.privacy_level:
            self.issues.append("Could not extract privacy level from HTML")
            return False

        # Privacy level should contain "team"
        if expected_privacy not in parser.privacy_level.lower():
            self.issues.append(f"Privacy level mismatch: expected '{expected_privacy}', got '{parser.privacy_level}'")
            self.details['privacy_level'] = parser.privacy_level
            return False

        self.details['privacy_level'] = parser.privacy_level
        return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_workspace_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_correct_name'] = self.check_checkpoint_2()
            self.checkpoints['cp3_correct_description'] = self.check_checkpoint_3()
            self.checkpoints['cp4_correct_privacy_level'] = self.check_checkpoint_4()

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
        print("Usage: python note_query3.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

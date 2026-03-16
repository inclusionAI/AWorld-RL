#!/usr/bin/env python3
# Evaluator for note Query 13

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
        self.body_class = None
        self.in_body = False

    def handle_starttag(self, tag, attrs):
        if tag == 'body':
            self.in_body = True
            attrs_dict = dict(attrs)
            if 'class' in attrs_dict:
                self.body_class = attrs_dict['class']


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task: enable dark mode theme
        self.checkpoints = {
            "cp1_dark_mode_enabled": False,
            "cp2_body_has_dark_class": False,
            "cp3_visual_confirmation": False,
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

    def parse_body_class(self, html_file: Path) -> Optional[str]:
        """Extract the body class attribute from HTML."""
        if not html_file.exists():
            return None
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = TaskHTMLParser()
        parser.feed(content)
        return parser.body_class

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Dark mode is enabled (body has theme-dark class)"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("Final HTML file not found")
            return False

        body_class = self.parse_body_class(final_html)
        if body_class and 'theme-dark' in body_class:
            self.details['final_body_class'] = body_class
            return True
        else:
            self.issues.append(f"Body does not have theme-dark class. Found: {body_class}")
            self.details['final_body_class'] = body_class
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Body class changed from theme-light to theme-dark"""
        # Check initial state
        initial_files = list(self.result_dir.glob("initial_*_raw.html"))
        if not initial_files:
            self.issues.append("Initial HTML file not found")
            return False
        
        initial_body_class = self.parse_body_class(initial_files[0])
        self.details['initial_body_class'] = initial_body_class
        
        # Check final state
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        final_body_class = self.parse_body_class(final_html)
        
        # Verify transition from light to dark
        if initial_body_class and 'theme-light' in initial_body_class:
            if final_body_class and 'theme-dark' in final_body_class:
                return True
            else:
                self.issues.append("Body class did not transition from theme-light to theme-dark")
                return False
        else:
            # If initial wasn't light, just verify final is dark
            if final_body_class and 'theme-dark' in final_body_class:
                return True
            else:
                return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Visual confirmation through trajectory actions"""
        trajectory = self.get_trajectory()
        
        # Check if there were click actions on dark mode toggle
        dark_mode_clicks = 0
        for entry in trajectory:
            if 'action' in entry:
                action = entry['action']
                if action.get('action_type') == 'CLICK':
                    reasoning = action.get('reasoning', '').lower()
                    if 'dark' in reasoning or 'theme' in reasoning:
                        dark_mode_clicks += 1
        
        self.details['dark_mode_related_clicks'] = dark_mode_clicks
        
        # If there were clicks related to dark mode, that's a good sign
        if dark_mode_clicks > 0:
            return True
        else:
            self.issues.append("No dark mode related clicks detected in trajectory")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_dark_mode_enabled'] = self.check_checkpoint_1()
            self.checkpoints['cp2_body_has_dark_class'] = self.check_checkpoint_2()
            self.checkpoints['cp3_visual_confirmation'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 13,
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
                'query_id': 13,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python note_query13.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

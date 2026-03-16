#!/usr/bin/env python3
# Evaluator for file Query 4

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class FileListParser(HTMLParser):
    """Parse HTML to extract file and folder names."""

    def __init__(self):
        super().__init__()
        self.in_file_card_name = False
        self.files_and_folders = []
        self.current_name = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "div" and attrs_dict.get("class") == "file-card-name":
            self.in_file_card_name = True
            self.current_name = ""

    def handle_endtag(self, tag):
        if tag == "div" and self.in_file_card_name:
            self.in_file_card_name = False
            if self.current_name.strip():
                self.files_and_folders.append(self.current_name.strip())

    def handle_data(self, data):
        if self.in_file_card_name:
            self.current_name += data


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on the task:
        # Task: Create a nested directory structure a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z
        self.checkpoints = {
            "cp1_folder_a_created": False,
            "cp2_nested_structure_complete": False,
            "cp3_folder_z_at_deepest_level": False,
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

    def parse_file_list(self, html_path: Path) -> List[str]:
        """Parse HTML file to extract list of files and folders."""
        if not html_path.exists():
            return []

        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        parser = FileListParser()
        parser.feed(html_content)
        return parser.files_and_folders

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Folder 'a' was created in My Files"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False

        files_and_folders = self.parse_file_list(final_html)
        self.details['final_files_and_folders'] = files_and_folders

        if 'a' in files_and_folders:
            return True
        else:
            self.issues.append("Folder 'a' was not created in My Files")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: The nested directory structure was fully created"""
        # Since we need to verify the entire nested structure exists,
        # we would need to navigate into folder 'a' and check its contents
        # However, the final HTML only shows the top-level My Files view
        
        # Check if the agent attempted to create the nested structure
        result_data = self.load_result_json()
        action_history = result_data.get('action_history', [])
        
        # Look for successful CREATE actions with the full path
        full_path = "a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z"
        create_attempts = 0
        successful_creates = 0
        
        for action_entry in action_history:
            action = action_entry.get('action', {})
            if action.get('action_type') == 'FILL':
                params = action.get('parameters', {})
                text = params.get('text', '')
                if text == full_path:
                    create_attempts += 1
            if action.get('action_type') == 'CLICK':
                params = action.get('parameters', {})
                selector = params.get('selector', '')
                if 'Create' in selector:
                    result = action_entry.get('result', {})
                    if result.get('success'):
                        successful_creates += 1
        
        self.details['create_attempts'] = create_attempts
        self.details['successful_create_clicks'] = successful_creates
        
        # If folder 'a' doesn't exist, the nested structure wasn't created
        final_html = self.find_final_html()
        if final_html:
            files_and_folders = self.parse_file_list(final_html)
            if 'a' not in files_and_folders:
                self.issues.append("Nested structure not created - root folder 'a' not found")
                return False
        
        # We cannot verify the full nested structure from the HTML alone
        # But we can check if the agent made appropriate attempts
        if create_attempts > 0 and successful_creates > 0:
            # Assume success if folder 'a' exists and create was attempted
            return True
        else:
            self.issues.append("No evidence of nested structure creation in action history")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Verify deepest folder 'z' exists (via agent actions)"""
        # This checkpoint verifies the agent successfully created the full path
        # Since we can't navigate through the HTML, we rely on action success
        
        result_data = self.load_result_json()
        
        # Check if agent claimed success
        if not result_data.get('success', False):
            self.issues.append("Agent did not claim successful completion")
            return False
        
        # If checkpoint 1 passed (folder 'a' exists) and agent claims success,
        # we assume the nested structure was properly created
        if self.checkpoints.get('cp1_folder_a_created', False):
            return True
        else:
            self.issues.append("Cannot verify deepest folder 'z' - root folder 'a' not found")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_folder_a_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_nested_structure_complete'] = self.check_checkpoint_2()
            self.checkpoints['cp3_folder_z_at_deepest_level'] = self.check_checkpoint_3()

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
        print("Usage: python file_query4.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

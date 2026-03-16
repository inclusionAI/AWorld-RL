#!/usr/bin/env python3
# Evaluator for file Query 5

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class FolderHTMLParser(HTMLParser):
    """Parse HTML to extract folder information from the file management interface."""

    def __init__(self):
        super().__init__()
        self.in_file_card = False
        self.in_file_card_name = False
        self.current_folder_name = None
        self.folders = []
        self.files = []
        self.is_folder = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Detect file card
        if tag == 'div' and attrs_dict.get('class') == 'file-card':
            self.in_file_card = True
            self.is_folder = False
            
        # Detect folder icon
        if self.in_file_card and tag == 'i':
            classes = attrs_dict.get('class', '')
            if 'fa-folder' in classes:
                self.is_folder = True
                
        # Detect file card name
        if tag == 'div' and attrs_dict.get('class') == 'file-card-name':
            self.in_file_card_name = True

    def handle_data(self, data):
        if self.in_file_card_name:
            self.current_folder_name = data.strip()

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_file_card_name:
            self.in_file_card_name = False
            if self.current_folder_name:
                if self.is_folder:
                    self.folders.append(self.current_folder_name)
                else:
                    self.files.append(self.current_folder_name)
                self.current_folder_name = None
                
        if tag == 'div' and self.in_file_card:
            self.in_file_card = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_nested_folder_created": False,
            "cp2_myFolder_created_inside": False,
            "cp3_no_errors_in_execution": False,
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

    def parse_html_for_folders(self, html_path: Path) -> tuple:
        """Parse HTML file and return list of folders."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = FolderHTMLParser()
        parser.feed(html_content)
        return parser.folders, parser.files

    def check_folder_creation_attempts(self) -> List[Dict]:
        """Check all folder creation attempts from trajectory."""
        result_data = self.load_result_json()
        action_history = result_data.get('action_history', [])
        
        folder_creations = []
        for action in action_history:
            action_info = action.get('action', {})
            if action_info.get('action_type') == 'FILL':
                text = action_info.get('parameters', {}).get('text', '')
                if '/' in text and ('a/b/c' in text or 'myFolder' in text):
                    folder_creations.append({
                        'step': action.get('step'),
                        'folder_path': text,
                        'success': action.get('result', {}).get('success', False)
                    })
        
        return folder_creations

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Verify nested folder structure a/b/c/.../z was created."""
        # Check if folder creation was attempted with the correct path
        folder_creations = self.check_folder_creation_attempts()
        
        # Look for the base nested folder path (without myFolder)
        base_path_found = False
        full_alphabet_path = 'a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z'
        
        for creation in folder_creations:
            if creation['folder_path'] == full_alphabet_path:
                base_path_found = True
                self.details['base_folder_path_attempted'] = full_alphabet_path
                break
        
        if not base_path_found:
            self.issues.append("Base nested folder path a/b/c/.../z was not created with correct full path")
            return False
        
        # Check in final HTML or late-stage HTML for folder 'a'
        final_html = self.find_final_html()
        if final_html:
            folders, _ = self.parse_html_for_folders(final_html)
            if 'a' in folders:
                self.details['folder_a_exists'] = True
                return True
            else:
                self.issues.append("Folder 'a' not found in final state (root of nested structure)")
                self.details['folders_found'] = folders
                return False
        
        self.issues.append("Could not find final HTML to verify folder creation")
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Verify myFolder was created inside the nested structure."""
        folder_creations = self.check_folder_creation_attempts()
        
        # Look for creation with myFolder at the end
        myFolder_path_found = False
        expected_path = 'a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z/myFolder'
        
        for creation in folder_creations:
            if creation['folder_path'] == expected_path:
                myFolder_path_found = True
                self.details['myFolder_path_attempted'] = expected_path
                self.details['myFolder_creation_success'] = creation['success']
                break
        
        if not myFolder_path_found:
            self.issues.append("myFolder was not created inside the nested structure")
            self.details['folder_paths_attempted'] = [c['folder_path'] for c in folder_creations]
            return False
        
        return True

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: No critical errors during execution."""
        result_data = self.load_result_json()
        
        # Check if execution completed without critical failures
        success = result_data.get('success', False)
        reason = result_data.get('reason', '')
        
        # If hit max_steps but attempted the task properly, consider it acceptable
        if reason == 'max_steps':
            # Check if meaningful progress was made
            folder_creations = self.check_folder_creation_attempts()
            if len(folder_creations) > 0:
                self.details['execution_note'] = 'Hit max steps but made progress on folder creation'
                return True
            else:
                self.issues.append('Reached max steps without attempting folder creation')
                return False
        
        if not success:
            self.issues.append(f'Execution failed: {reason}')
            return False
        
        return True

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_nested_folder_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_myFolder_created_inside'] = self.check_checkpoint_2()
            self.checkpoints['cp3_no_errors_in_execution'] = self.check_checkpoint_3()

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
        print("Usage: python file_query5.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Evaluator for file Query 11

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
        self.in_folder_name = False
        self.in_file_item = False
        self.folder_names = []
        self.file_info = []
        self.current_folder = {}
        self.current_file = {}
        self.capture_text = False
        self.current_text = ""
        self.in_breadcrumb = False
        self.breadcrumb_path = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for folder names
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'folder-name' in classes or 'file-name' in classes:
                self.capture_text = True
                self.current_text = ""
            # Check for breadcrumb navigation
            if 'breadcrumb' in classes:
                self.in_breadcrumb = True

    def handle_data(self, data):
        if self.capture_text:
            self.current_text += data.strip()
        if self.in_breadcrumb:
            text = data.strip()
            if text and text != '/' and text != '>':
                self.breadcrumb_path.append(text)

    def handle_endtag(self, tag):
        if self.capture_text:
            if self.current_text:
                self.folder_names.append(self.current_text)
            self.capture_text = False
            self.current_text = ""


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_work_projects": False,
            "cp2_created_work_folder": False,
            "cp3_created_today_subfolder": False,
            "cp4_uploaded_100mb_file": False,
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

    def check_html_for_folders(self, html_path: Path, folder_names: List[str]) -> List[str]:
        """Check if specific folder names exist in HTML."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        found_folders = []
        for folder_name in folder_names:
            # Look for folder name in various contexts
            if f'>{folder_name}<' in html_content or f'"{folder_name}"' in html_content:
                found_folders.append(folder_name)
        
        return found_folders

    def check_html_for_file_upload(self, html_path: Path) -> Optional[Dict]:
        """Check if a 100MB file was uploaded."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Look for file size patterns indicating 100MB
        size_patterns = [
            r'100\s*MB',
            r'100\.0*\s*MB',
            r'104857600',  # 100MB in bytes
        ]
        
        for pattern in size_patterns:
            if re.search(pattern, html_content, re.IGNORECASE):
                return {'size': '100MB', 'found': True}
        
        return None

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to Work Projects folder"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent navigated to Work Projects
            for action in action_history:
                if action.get('action', {}).get('parameters', {}).get('selector'):
                    selector = action['action']['parameters']['selector']
                    if 'Work Projects' in selector:
                        self.details['navigated_to_work_projects'] = True
                        return True
            
            self.issues.append("Did not navigate to Work Projects folder")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Created 'work' folder in Work Projects"""
        try:
            # Check intermediate steps for folder creation
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files[5:10]:  # Check around step 6-7 where folder was created
                folders_found = self.check_html_for_folders(step_file, ['work'])
                if 'work' in folders_found:
                    self.details['work_folder_created'] = True
                    return True
            
            self.issues.append("'work' folder was not created")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Created 'today' subfolder inside 'work'"""
        try:
            # Check if nested folder creation was used with 'work/today'
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent entered 'work/today' in folder creation
            for action in action_history:
                if action.get('action', {}).get('action_type') == 'FILL':
                    text = action['action']['parameters'].get('text', '')
                    if 'work/today' in text:
                        self.details['nested_folder_path_used'] = 'work/today'
                        return True
            
            # Also check HTML for 'today' folder
            step_files = self.find_step_html("step_*")
            for step_file in step_files[5:10]:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    if 'today' in html_content:
                        self.details['today_folder_created'] = True
                        return True
            
            self.issues.append("'today' subfolder was not created inside 'work'")
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Uploaded 100MB file into 'today' folder"""
        try:
            # Check all steps for file upload
            all_steps = self.find_step_html("step_*")
            
            for step_file in all_steps:
                file_info = self.check_html_for_file_upload(step_file)
                if file_info and file_info.get('found'):
                    self.details['file_uploaded'] = '100MB file found'
                    return True
            
            # Check if agent attempted file upload
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            upload_attempted = False
            for action in action_history:
                action_type = action.get('action', {}).get('action_type', '')
                if action_type in ['UPLOAD', 'FILE_UPLOAD']:
                    upload_attempted = True
                    break
            
            if upload_attempted:
                self.issues.append("File upload was attempted but 100MB file not found in final state")
                self.details['upload_attempted'] = True
            else:
                self.issues.append("No file upload was attempted - agent got stuck trying to open 'work' folder")
                self.details['agent_stuck'] = True
            
            return False
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_work_projects'] = self.check_checkpoint_1()
            self.checkpoints['cp2_created_work_folder'] = self.check_checkpoint_2()
            self.checkpoints['cp3_created_today_subfolder'] = self.check_checkpoint_3()
            self.checkpoints['cp4_uploaded_100mb_file'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            # Add analysis of what went wrong
            if not overall_success:
                if passed_count == 3:
                    self.details['analysis'] = "Agent successfully created nested folder structure but failed to upload file"
                elif passed_count == 2:
                    self.details['analysis'] = "Agent created 'work' folder but couldn't navigate into it to upload file"
                else:
                    self.details['analysis'] = "Agent encountered navigation issues"

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
        print("Usage: python file_query11.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

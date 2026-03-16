#!/usr/bin/env python3
# Evaluator for note Query 4

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract workspace and folder information."""

    def __init__(self):
        super().__init__()
        self.in_workspace_title = False
        self.in_folder_name = False
        self.in_stat_value = False
        self.in_stat_label = False
        self.workspace_name = None
        self.folder_names = []
        self.current_folder_name = None
        self.stat_labels = []
        self.stat_values = []
        self.current_stat_label = None
        self.current_stat_value = None
        self.tag_stack = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.tag_stack.append((tag, attrs_dict))
        
        # Check for workspace title
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'workspace-title' in classes:
                self.in_workspace_title = True
            elif 'folder-name' in classes:
                self.in_folder_name = True
            elif 'stat-value' in classes:
                self.in_stat_value = True
            elif 'stat-label' in classes:
                self.in_stat_label = True

    def handle_endtag(self, tag):
        if self.tag_stack:
            self.tag_stack.pop()
        
        if self.in_workspace_title:
            self.in_workspace_title = False
        if self.in_folder_name:
            if self.current_folder_name:
                self.folder_names.append(self.current_folder_name)
                self.current_folder_name = None
            self.in_folder_name = False
        if self.in_stat_value:
            if self.current_stat_value:
                self.stat_values.append(self.current_stat_value)
                self.current_stat_value = None
            self.in_stat_value = False
        if self.in_stat_label:
            if self.current_stat_label:
                self.stat_labels.append(self.current_stat_label)
                self.current_stat_label = None
            self.in_stat_label = False

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
            
        if self.in_workspace_title:
            self.workspace_name = text
        elif self.in_folder_name:
            self.current_folder_name = text
        elif self.in_stat_value:
            self.current_stat_value = text
        elif self.in_stat_label:
            self.current_stat_label = text


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_workspace_page": False,
            "cp2_navigated_to_personal_workspace": False,
            "cp3_folder_created_successfully": False,
            "cp4_folder_name_correct": False,
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
        """Checkpoint 1: Agent navigated to workspace page."""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent reached workspace page
            for step in trajectory:
                page_info = step.get('page_info', {})
                url = page_info.get('url', '')
                if '/workspace' in url:
                    self.details['workspace_page_reached'] = True
                    return True
            
            self.issues.append("Agent did not navigate to workspace page")
            self.details['workspace_page_reached'] = False
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking workspace navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Agent navigated to Personal Workspace."""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent reached Personal Workspace (workspaces/500)
            for step in trajectory:
                page_info = step.get('page_info', {})
                url = page_info.get('url', '')
                if '/workspaces/500' in url:
                    self.details['personal_workspace_reached'] = True
                    return True
            
            self.issues.append("Agent did not navigate to Personal Workspace")
            self.details['personal_workspace_reached'] = False
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Personal Workspace navigation: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Folder was created successfully."""
        try:
            final_html_file = self.find_final_html()
            if not final_html_file:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TaskHTMLParser()
            parser.feed(html_content)

            # Check if we're in Personal Workspace
            if parser.workspace_name != 'Personal Workspace':
                self.issues.append(f"Not in Personal Workspace. Found: {parser.workspace_name}")
                self.details['final_workspace'] = parser.workspace_name
                return False

            # Check folder count in stats
            folder_count = None
            for i, label in enumerate(parser.stat_labels):
                if label == 'Folders':
                    if i < len(parser.stat_values):
                        folder_count = parser.stat_values[i]
                        break

            if folder_count == '1':
                self.details['folder_count'] = folder_count
                self.details['folders_found'] = parser.folder_names
                return True
            else:
                self.issues.append(f"Folder count is {folder_count}, expected 1")
                self.details['folder_count'] = folder_count
                return False

        except Exception as e:
            self.issues.append(f"Error checking folder creation: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Folder name is correct (Sprint Documents)."""
        try:
            final_html_file = self.find_final_html()
            if not final_html_file:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TaskHTMLParser()
            parser.feed(html_content)

            # Check if Sprint Documents folder exists
            if 'Sprint Documents' in parser.folder_names:
                self.details['folder_name_correct'] = True
                return True
            
            # Also check for truncated version in HTML
            if any('Sprint Docum' in name for name in parser.folder_names):
                # Verify in raw HTML
                if 'Sprint Documents' in html_content:
                    self.details['folder_name_correct'] = True
                    self.details['folder_name_note'] = 'Name found in HTML (may be truncated in display)'
                    return True

            self.issues.append(f"Folder name 'Sprint Documents' not found. Found: {parser.folder_names}")
            self.details['folder_names_found'] = parser.folder_names
            return False

        except Exception as e:
            self.issues.append(f"Error checking folder name: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_workspace_page'] = self.check_checkpoint_1()
            self.checkpoints['cp2_navigated_to_personal_workspace'] = self.check_checkpoint_2()
            self.checkpoints['cp3_folder_created_successfully'] = self.check_checkpoint_3()
            self.checkpoints['cp4_folder_name_correct'] = self.check_checkpoint_4()

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
        print("Usage: python note_query4.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

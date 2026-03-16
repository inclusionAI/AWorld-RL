#!/usr/bin/env python3
# Evaluator for file Query 3

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class FolderStructureParser(HTMLParser):
    """Parse HTML to extract folder structure information."""

    def __init__(self):
        super().__init__()
        self.folders = []
        self.current_folder = None
        self.in_folder_item = False
        self.in_folder_name = False
        self.current_text = ""
        self.folder_hierarchy = {}
        self.current_attrs = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Look for folder items
        if tag == "div" and any("folder" in str(v).lower() for k, v in attrs):
            self.in_folder_item = True
            self.current_attrs = attrs_dict
            self.current_text = ""
        
        # Check for folder names in various structures
        if self.in_folder_item and tag in ["div", "span", "p"]:
            self.in_folder_name = True

    def handle_data(self, data):
        data = data.strip()
        if data and self.in_folder_item:
            self.current_text += data + " "
            
    def handle_endtag(self, tag):
        if tag == "div" and self.in_folder_item:
            text = self.current_text.strip()
            if text and ("Personal" in text or "Archive" in text or "Work" in text or "Projects" in text):
                self.folders.append(text)
            self.in_folder_item = False
            self.in_folder_name = False
            self.current_text = ""
            self.current_attrs = {}


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on Query 3:
        # "Rename 'Personal Files' to 'Personal Archive' and move it under 'Work Projects'"
        self.checkpoints = {
            "cp1_personal_files_renamed": False,
            "cp2_personal_archive_exists": False,
            "cp3_moved_under_work_projects": False,
            "cp4_original_personal_files_gone": False,
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

    def parse_html_for_folders(self, html_path: Path) -> Dict[str, any]:
        """Parse HTML to extract folder information."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Look for folder names in the HTML
        personal_archive_found = bool(re.search(r'Personal\s+Archive', html_content, re.IGNORECASE))
        personal_files_found = bool(re.search(r'Personal\s+Files', html_content, re.IGNORECASE))
        work_projects_found = bool(re.search(r'Work\s+Projects', html_content, re.IGNORECASE))
        
        # Try to determine folder hierarchy
        # Look for nested structure patterns
        hierarchy_info = self.extract_folder_hierarchy(html_content)
        
        return {
            'personal_archive_found': personal_archive_found,
            'personal_files_found': personal_files_found,
            'work_projects_found': work_projects_found,
            'hierarchy': hierarchy_info
        }

    def extract_folder_hierarchy(self, html_content: str) -> Dict:
        """Extract folder hierarchy from HTML content."""
        hierarchy = {}
        
        # Look for patterns that indicate Personal Archive is under Work Projects
        # This is a heuristic approach - look for these folders appearing close together
        # in the DOM structure
        
        # Split by common separators and look for folder relationships
        lines = html_content.split('\n')
        
        work_projects_line = -1
        personal_archive_line = -1
        personal_files_line = -1
        
        for i, line in enumerate(lines):
            if re.search(r'Work\s+Projects', line, re.IGNORECASE):
                work_projects_line = i
            if re.search(r'Personal\s+Archive', line, re.IGNORECASE):
                personal_archive_line = i
            if re.search(r'Personal\s+Files', line, re.IGNORECASE):
                personal_files_line = i
        
        # If Personal Archive appears shortly after Work Projects in DOM,
        # it's likely nested under it
        hierarchy['work_projects_line'] = work_projects_line
        hierarchy['personal_archive_line'] = personal_archive_line
        hierarchy['personal_files_line'] = personal_files_line
        
        # Check if Personal Archive is likely under Work Projects
        # (appears within reasonable distance in DOM after Work Projects)
        if work_projects_line > 0 and personal_archive_line > 0:
            distance = personal_archive_line - work_projects_line
            hierarchy['likely_nested'] = 0 < distance < 100
        else:
            hierarchy['likely_nested'] = False
            
        return hierarchy

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Personal Files was renamed to Personal Archive."""
        html_file = self.find_final_html()
        if not html_file:
            self.issues.append("No final HTML file found")
            return False
        
        folder_info = self.parse_html_for_folders(html_file)
        
        # Check if "Personal Archive" exists
        if folder_info['personal_archive_found']:
            self.details['personal_archive_found'] = True
            return True
        else:
            self.issues.append("'Personal Archive' folder not found in final state")
            self.details['personal_archive_found'] = False
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Personal Archive folder exists in the system."""
        html_file = self.find_final_html()
        if not html_file:
            return False
        
        folder_info = self.parse_html_for_folders(html_file)
        
        if folder_info['personal_archive_found']:
            self.details['personal_archive_exists'] = True
            return True
        else:
            self.issues.append("Personal Archive folder does not exist")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Personal Archive is moved under Work Projects."""
        html_file = self.find_final_html()
        if not html_file:
            return False
        
        folder_info = self.parse_html_for_folders(html_file)
        hierarchy = folder_info['hierarchy']
        
        # Check if folder appears to be nested
        if hierarchy.get('likely_nested', False):
            self.details['moved_under_work_projects'] = True
            self.details['hierarchy_info'] = f"Personal Archive found near Work Projects (distance: {hierarchy.get('personal_archive_line', 0) - hierarchy.get('work_projects_line', 0)} lines)"
            return True
        else:
            self.issues.append("Personal Archive does not appear to be under Work Projects")
            self.details['moved_under_work_projects'] = False
            
            # Provide diagnostic info
            if hierarchy.get('work_projects_line', -1) == -1:
                self.details['work_projects_found'] = False
            if hierarchy.get('personal_archive_line', -1) == -1:
                self.details['personal_archive_found'] = False
                
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Original 'Personal Files' folder no longer exists (was renamed)."""
        html_file = self.find_final_html()
        if not html_file:
            return False
        
        folder_info = self.parse_html_for_folders(html_file)
        
        # The original "Personal Files" should NOT exist if it was successfully renamed
        if not folder_info['personal_files_found']:
            self.details['original_personal_files_removed'] = True
            return True
        else:
            self.issues.append("Original 'Personal Files' folder still exists (was not renamed)")
            self.details['original_personal_files_removed'] = False
            self.details['personal_files_still_exists'] = True
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_personal_files_renamed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_personal_archive_exists'] = self.check_checkpoint_2()
            self.checkpoints['cp3_moved_under_work_projects'] = self.check_checkpoint_3()
            self.checkpoints['cp4_original_personal_files_gone'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 3,
                'query': "Rename 'Personal Files' to 'Personal Archive' and move it under 'Work Projects'",
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
        print("Usage: python file_query3.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

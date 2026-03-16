#!/usr/bin/env python3
# Evaluator for file Query 6

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TrashHTMLParser(HTMLParser):
    """Parse HTML to extract trash and file information."""

    def __init__(self):
        super().__init__()
        self.in_trash_main = False
        self.in_empty_state = False
        self.in_trash_list = False
        self.trash_is_empty = False
        self.trash_files = []
        self.current_file_name = None
        self.in_file_name = False
        self.in_myfiles_main = False
        self.myfiles_files = []
        self.in_file_card_name = False
        self.current_myfiles_file = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get('class', '')
        
        if 'trash-main' in class_attr:
            self.in_trash_main = True
        elif 'empty-state' in class_attr and self.in_trash_main:
            self.in_empty_state = True
        elif 'trash-list' in class_attr:
            self.in_trash_list = True
        elif 'trash-item-name' in class_attr and self.in_trash_list:
            self.in_file_name = True
        elif 'myfiles-main' in class_attr:
            self.in_myfiles_main = True
        elif 'file-card-name' in class_attr and self.in_myfiles_main:
            self.in_file_card_name = True

    def handle_endtag(self, tag):
        if tag == 'main':
            self.in_trash_main = False
            self.in_myfiles_main = False
        elif tag == 'div':
            if self.in_empty_state:
                self.in_empty_state = False
            if self.in_file_name:
                self.in_file_name = False
            if self.in_file_card_name:
                self.in_file_card_name = False

    def handle_data(self, data):
        text = data.strip()
        if self.in_empty_state and 'Trash is empty' in text:
            self.trash_is_empty = True
        elif self.in_file_name and text:
            self.current_file_name = text
            self.trash_files.append(text)
        elif self.in_file_card_name and text:
            self.current_myfiles_file = text
            self.myfiles_files.append(text)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints for this task
        self.checkpoints = {
            "cp1_navigated_to_myfiles": False,
            "cp2_deleted_welcome_guide": False,
            "cp3_navigated_to_trash": False,
            "cp4_file_appears_in_trash": False,
            "cp5_file_restored": False,
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
                if line.strip():
                    trajectory.append(json.loads(line))
        return trajectory

    def load_result_json(self) -> Dict:
        """Load result.json file."""
        if not self.result_file.exists():
            raise FileNotFoundError(f"result.json not found")

        with open(self.result_file, 'r') as f:
            return json.load(f)

    def parse_html(self, html_path: Path) -> TrashHTMLParser:
        """Parse HTML file and return parser with extracted data."""
        parser = TrashHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Agent navigated to My Files."""
        trajectory = self.get_trajectory()
        
        for entry in trajectory:
            page_info = entry.get('page_info', {})
            url = page_info.get('url', '')
            if '/my-files' in url:
                self.details['myfiles_navigation'] = 'Agent successfully navigated to My Files'
                return True
        
        self.issues.append("Agent did not navigate to My Files")
        return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: welcome_guide.pdf was deleted from My Files."""
        # Check initial state - file should exist
        step_files = self.find_step_html("step_*")
        if not step_files:
            self.issues.append("No step files found to verify deletion")
            return False
        
        # Find a step in My Files before deletion
        initial_has_file = False
        for step_file in step_files[:3]:  # Check early steps
            parser = self.parse_html(step_file)
            if 'welcome_guide.pdf' in parser.myfiles_files:
                initial_has_file = True
                break
        
        if not initial_has_file:
            self.issues.append("welcome_guide.pdf was not present initially in My Files")
            return False
        
        # Check after deletion - look for steps after trash navigation
        trajectory = self.get_trajectory()
        trash_step = None
        for entry in trajectory:
            if '/trash' in entry.get('page_info', {}).get('url', ''):
                trash_step = entry.get('step_num')
                break
        
        # Check a My Files page after trash navigation (if agent went back)
        # Or check right before trash navigation
        deleted = False
        for step_file in step_files[3:]:
            parser = self.parse_html(step_file)
            if parser.in_myfiles_main and 'welcome_guide.pdf' not in parser.myfiles_files:
                deleted = True
                self.details['file_deletion'] = 'welcome_guide.pdf was deleted from My Files'
                break
        
        # Check trajectory for DELETE action
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'CLICK':
                selector = action.get('parameters', {}).get('selector', '')
                if 'Delete' in selector or 'delete' in selector.lower():
                    deleted = True
                    self.details['delete_action'] = 'Agent clicked Delete in context menu'
                    break
        
        if not deleted:
            self.issues.append("welcome_guide.pdf was not deleted from My Files")
            return False
        
        return True

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Agent navigated to Trash."""
        trajectory = self.get_trajectory()
        
        for entry in trajectory:
            page_info = entry.get('page_info', {})
            url = page_info.get('url', '')
            if '/trash' in url:
                self.details['trash_navigation'] = 'Agent successfully navigated to Trash'
                return True
        
        self.issues.append("Agent did not navigate to Trash")
        return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: welcome_guide.pdf appears in Trash."""
        # Check all steps after trash navigation
        step_files = self.find_step_html("step_*")
        
        for step_file in step_files:
            parser = self.parse_html(step_file)
            if parser.in_trash_main and not parser.trash_is_empty:
                if 'welcome_guide.pdf' in parser.trash_files:
                    self.details['file_in_trash'] = 'welcome_guide.pdf found in Trash'
                    return True
        
        # Check final HTML
        final_html = self.find_final_html()
        if final_html:
            parser = self.parse_html(final_html)
            if parser.in_trash_main:
                if parser.trash_is_empty:
                    self.issues.append("Trash is empty - welcome_guide.pdf does not appear in Trash")
                    self.details['trash_status'] = 'Trash is empty'
                elif 'welcome_guide.pdf' in parser.trash_files:
                    self.details['file_in_trash'] = 'welcome_guide.pdf found in Trash'
                    return True
                else:
                    self.issues.append("welcome_guide.pdf not found in Trash")
                    self.details['trash_files'] = parser.trash_files
        else:
            self.issues.append("No final HTML found to check Trash contents")
        
        return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: File was restored from Trash."""
        # Check trajectory for restore action
        trajectory = self.get_trajectory()
        
        for entry in trajectory:
            action = entry.get('action', {})
            if action.get('action_type') == 'CLICK':
                selector = action.get('parameters', {}).get('selector', '')
                if 'Restore' in selector or 'restore' in selector.lower():
                    self.details['restore_action'] = 'Agent clicked Restore'
                    
                    # Verify file is back in My Files
                    step_files = self.find_step_html("step_*")
                    for step_file in step_files[-3:]:  # Check last few steps
                        parser = self.parse_html(step_file)
                        if parser.in_myfiles_main and 'welcome_guide.pdf' in parser.myfiles_files:
                            self.details['file_restored'] = 'welcome_guide.pdf restored to My Files'
                            return True
                    break
        
        # Check final state - file should be in My Files and not in Trash
        final_html = self.find_final_html()
        if final_html:
            parser = self.parse_html(final_html)
            if parser.in_myfiles_main and 'welcome_guide.pdf' in parser.myfiles_files:
                self.details['file_restored'] = 'welcome_guide.pdf is in My Files'
                return True
            elif parser.in_trash_main and 'welcome_guide.pdf' not in parser.trash_files:
                # File not in trash but also need to verify it's back in My Files
                pass
        
        self.issues.append("File was not restored from Trash")
        return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_myfiles'] = self.check_checkpoint_1()
            self.checkpoints['cp2_deleted_welcome_guide'] = self.check_checkpoint_2()
            self.checkpoints['cp3_navigated_to_trash'] = self.check_checkpoint_3()
            self.checkpoints['cp4_file_appears_in_trash'] = self.check_checkpoint_4()
            self.checkpoints['cp5_file_restored'] = self.check_checkpoint_5()

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
        print("Usage: python file_query6.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Evaluator for file Query 7

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class FileListParser(HTMLParser):
    """Parse HTML to extract file information from search results."""

    def __init__(self):
        super().__init__()
        self.files = []
        self.current_file = {}
        self.in_file_name = False
        self.in_file_size = False
        self.in_result_item = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and attrs_dict.get('class') == 'result-item':
            self.in_result_item = True
            self.current_file = {}
            
        if self.in_result_item:
            if tag == 'h3' and 'file-name' in attrs_dict.get('class', ''):
                self.in_file_name = True
            elif tag == 'span' and 'file-size' in attrs_dict.get('class', ''):
                self.in_file_size = True

    def handle_data(self, data):
        if self.in_file_name:
            self.current_file['name'] = data.strip()
            self.in_file_name = False
        elif self.in_file_size:
            self.current_file['size'] = data.strip()
            self.in_file_size = False
            
    def handle_endtag(self, tag):
        if tag == 'div' and self.in_result_item and 'name' in self.current_file and 'size' in self.current_file:
            self.files.append(self.current_file.copy())
            self.in_result_item = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_search_executed": False,
            "cp2_files_identified": False,
            "cp3_files_deleted": False,
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

    def parse_file_size(self, size_str: str) -> float:
        """Convert file size string to MB."""
        size_str = size_str.strip().upper()
        if 'KB' in size_str:
            return float(size_str.replace('KB', '').strip()) / 1024
        elif 'MB' in size_str:
            return float(size_str.replace('MB', '').strip())
        elif 'GB' in size_str:
            return float(size_str.replace('GB', '').strip()) * 1024
        return 0.0

    def parse_files_from_html(self, html_path: Path) -> List[Dict]:
        """Parse file list from HTML."""
        if not html_path.exists():
            return []
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = FileListParser()
        parser.feed(html_content)
        return parser.files

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Search for 'welcome' was executed."""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Check if search was executed
            search_found = False
            for action in actions:
                action_obj = action.get('action', {})
                if action_obj.get('action_type') == 'FILL':
                    params = action_obj.get('parameters', {})
                    if 'search' in params.get('selector', '').lower():
                        if params.get('text', '').lower() == 'welcome':
                            search_found = True
                            
                if action_obj.get('action_type') == 'CLICK':
                    params = action_obj.get('parameters', {})
                    selector = params.get('selector', '')
                    if 'search' in selector.lower() and search_found:
                        self.details['search_executed'] = True
                        return True
            
            self.issues.append("Search for 'welcome' was not properly executed")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Files between 1MB and 10MB were identified."""
        try:
            # Find search results HTML (after search was executed)
            search_html_files = self.find_step_html("step_3_*")
            if not search_html_files:
                search_html_files = self.find_step_html("step_*")
            
            if not search_html_files:
                self.issues.append("No search results found")
                return False
            
            # Parse files from search results
            files = self.parse_files_from_html(search_html_files[0])
            
            target_files = []
            for file in files:
                if 'welcome' in file.get('name', '').lower():
                    size_mb = self.parse_file_size(file.get('size', '0'))
                    if 1.0 <= size_mb <= 10.0:
                        target_files.append({
                            'name': file['name'],
                            'size': file['size'],
                            'size_mb': size_mb
                        })
            
            self.details['files_in_range'] = target_files
            self.details['expected_file_count'] = len(target_files)
            
            if len(target_files) > 0:
                expected_names = [f['name'] for f in target_files]
                self.details['expected_files'] = expected_names
                return True
            else:
                self.issues.append("No files between 1MB and 10MB found in search results")
                return False
                
        except Exception as e:
            self.issues.append(f"Error identifying files: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Files between 1MB and 10MB were deleted."""
        try:
            # Parse files from final HTML
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("Final HTML state not found")
                return False
            
            final_files = self.parse_files_from_html(final_html)
            
            # Get expected files from checkpoint 2
            expected_files = self.details.get('expected_files', [])
            if not expected_files:
                self.issues.append("No expected files to check deletion")
                return False
            
            # Check if target files still exist
            final_file_names = [f.get('name', '') for f in final_files if 'welcome' in f.get('name', '').lower()]
            
            still_present = []
            for expected_file in expected_files:
                if expected_file in final_file_names:
                    still_present.append(expected_file)
            
            self.details['files_still_present'] = still_present
            self.details['final_welcome_files'] = final_file_names
            
            if len(still_present) == 0:
                self.details['deletion_success'] = True
                return True
            else:
                self.issues.append(f"Files not deleted: {', '.join(still_present)}")
                self.details['deletion_success'] = False
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking file deletion: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_search_executed'] = self.check_checkpoint_1()
            self.checkpoints['cp2_files_identified'] = self.check_checkpoint_2()
            self.checkpoints['cp3_files_deleted'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 7,
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
                'query_id': 7,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python file_query7.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

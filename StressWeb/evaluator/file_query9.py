#!/usr/bin/env python3
# Evaluator for file Query 9

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class FileDetailsParser(HTMLParser):
    """Parse HTML to extract file details from file detail pages."""

    def __init__(self):
        super().__init__()
        self.in_file_title = False
        self.current_file_name = None
        self.file_names_found = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'h1' and attrs_dict.get('class') == 'file-title':
            self.in_file_title = True

    def handle_data(self, data):
        if self.in_file_title:
            self.current_file_name = data.strip()
            if self.current_file_name and self.current_file_name.lower().endswith('.jpg'):
                self.file_names_found.append(self.current_file_name)

    def handle_endtag(self, tag):
        if tag == 'h1' and self.in_file_title:
            self.in_file_title = False


class PhotosDirectoryParser(HTMLParser):
    """Parse HTML to check if we're in the Photos directory and count JPG files."""

    def __init__(self):
        super().__init__()
        self.jpg_files = []
        self.in_file_name = False
        self.current_text = ""
        self.in_photos_dir = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        # Check for file name elements
        if tag == 'div' and attrs_dict.get('class') == 'file-name':
            self.in_file_name = True
            self.current_text = ""

    def handle_data(self, data):
        text = data.strip()
        if self.in_file_name:
            self.current_text += text
        # Check if we're in Photos directory
        if 'Photos' in text:
            self.in_photos_dir = True

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_file_name:
            self.in_file_name = False
            if self.current_text.lower().endswith('.jpg'):
                self.jpg_files.append(self.current_text)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints
        self.checkpoints = {
            "cp1_navigated_to_photos_dir": False,
            "cp2_identified_jpg_files": False,
            "cp3_all_jpg_files_deleted": False,
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
        """Checkpoint 1: Agent navigated to Personal Files/Photos directory"""
        try:
            # Check early steps (1-5) to see if agent reached Photos directory
            early_steps = self.find_step_html("step_[1-5]")
            
            for step_file in early_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    # Check if we're in Photos directory
                    if 'Photos' in html_content and ('Personal Files' in html_content or 'family_photo' in html_content or 'vacation' in html_content):
                        self.details['reached_photos_dir'] = True
                        return True
            
            self.issues.append("Agent did not navigate to Personal Files/Photos directory")
            return False
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Agent identified JPG files in Photos directory"""
        try:
            # Check steps 3-10 to see if agent found JPG files
            early_steps = self.find_step_html("step_[3-9]")
            
            jpg_files_found = set()
            for step_file in early_steps:
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    # Look for JPG file references
                    if 'family_photo_2025.jpg' in html_content or 'family_photo' in html_content:
                        jpg_files_found.add('family_photo_2025.jpg')
                    if 'vacation_2024.jpg' in html_content or 'vacation_2024' in html_content:
                        jpg_files_found.add('vacation_2024.jpg')
            
            self.details['jpg_files_identified'] = list(jpg_files_found)
            
            if len(jpg_files_found) > 0:
                return True
            else:
                self.issues.append("Agent did not identify JPG files in Photos directory")
                return False
        except Exception as e:
            self.issues.append(f"Error checking JPG file identification: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: All JPG files are deleted from Photos directory"""
        try:
            # Check the final HTML state
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML state found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Parse to see if we're still on a file detail page showing a JPG
            parser = FileDetailsParser()
            parser.feed(html_content)
            
            # If we're on a JPG file detail page, it means the file still exists
            if parser.file_names_found:
                jpg_file = parser.file_names_found[0]
                self.issues.append(f"JPG file '{jpg_file}' still exists and was not deleted")
                self.details['remaining_jpg_files'] = parser.file_names_found
                return False

            # Check if Photos directory shows no JPG files
            # The agent spent 100 steps clicking delete, which suggests the file wasn't actually being deleted
            # Looking at the action history, all 100 steps were clicking the same Delete button
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent was stuck in a loop
            delete_clicks = sum(1 for action in action_history 
                              if action.get('action', {}).get('action_type') == 'CLICK' 
                              and 'Delete' in str(action.get('action', {}).get('parameters', {})))
            
            if delete_clicks > 50:
                self.issues.append(f"Agent clicked Delete button {delete_clicks} times, suggesting the deletion was not successful")
                self.details['delete_clicks'] = delete_clicks
                return False

            # If we reached here without finding JPG files, check if deletion was actually successful
            # by looking at intermediate steps to see if JPG files disappeared
            last_10_steps = self.find_step_html("step_9[0-9]") + [final_html]
            
            for step_file in reversed(last_10_steps[-5:]):
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'family_photo_2025.jpg' in content:
                        self.issues.append("JPG files still present in final steps")
                        return False

            # If no JPG files found in final state, consider it successful
            self.details['deletion_status'] = 'completed'
            return True

        except Exception as e:
            self.issues.append(f"Error checking deletion status: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_photos_dir'] = self.check_checkpoint_1()
            self.checkpoints['cp2_identified_jpg_files'] = self.check_checkpoint_2()
            self.checkpoints['cp3_all_jpg_files_deleted'] = self.check_checkpoint_3()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 9,
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
                'query_id': 9,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python file_query9.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

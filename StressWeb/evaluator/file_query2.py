#!/usr/bin/env python3
# Evaluator for file Query 2

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class FileSystemHTMLParser(HTMLParser):
    """Parse HTML to extract file and folder information."""

    def __init__(self):
        super().__init__()
        self.in_file_list = False
        self.current_item = {}
        self.capture_text = False
        self.folders = []
        self.files = []
        self.current_tag = None
        self.folder_contents = {}  # Maps folder names to their contents
        self.current_path = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Look for folder or file items
        if tag == 'div' and 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'folder-item' in classes or 'file-item' in classes:
                self.current_item = {'type': 'folder' if 'folder-item' in classes else 'file', 'name': '', 'attrs': attrs_dict}
                self.capture_text = True
            elif 'folder-name' in classes or 'file-name' in classes:
                self.capture_text = True
                self.current_tag = 'name'

    def handle_data(self, data):
        if self.capture_text and data.strip():
            text = data.strip()
            if self.current_item:
                if 'name' not in self.current_item or not self.current_item['name']:
                    self.current_item['name'] = text
                elif self.current_tag == 'name':
                    self.current_item['name'] = text

    def handle_endtag(self, tag):
        if tag == 'div' and self.current_item and 'name' in self.current_item and self.current_item['name']:
            if self.current_item['type'] == 'folder':
                self.folders.append(self.current_item['name'])
            else:
                self.files.append(self.current_item['name'])
            self.current_item = {}
        self.capture_text = False
        self.current_tag = None


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_folder_created": False,
            "cp2_navigated_to_folder": False,
            "cp3_document_uploaded": False,
            "cp4_design_uploaded": False,
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

    def parse_html_for_items(self, html_path: Path) -> Tuple[List[str], List[str]]:
        """Parse HTML file and extract folders and files."""
        if not html_path.exists():
            return [], []
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parser = FileSystemHTMLParser()
        try:
            parser.feed(content)
        except Exception:
            pass
        
        return parser.folders, parser.files

    def search_in_html(self, html_path: Path, pattern: str) -> bool:
        """Search for a pattern in HTML file."""
        if not html_path.exists():
            return False
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return pattern in content

    def check_folder_in_intermediate_steps(self, folder_name: str) -> bool:
        """Check if folder was created in any intermediate step."""
        all_steps = self.find_step_html("step_*")
        for step_file in all_steps:
            if self.search_in_html(step_file, folder_name):
                return True
        return False

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Project Materials folder was created"""
        try:
            # Check if folder was created in any step
            folder_created = self.check_folder_in_intermediate_steps("Project Materials")
            
            if not folder_created:
                self.issues.append("Project Materials folder was not created")
                self.details['cp1_folder_created'] = "Folder not found in any step"
                return False
            
            self.details['cp1_folder_created'] = "Folder was created successfully"
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking folder creation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Agent navigated into the Project Materials folder"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if agent clicked on Project Materials folder
            for entry in trajectory:
                if 'action' in entry and 'parameters' in entry['action']:
                    params = entry['action']['parameters']
                    if 'selector' in params:
                        selector = params['selector']
                        if 'Project Materials' in selector:
                            self.details['cp2_navigated_to_folder'] = "Agent clicked on Project Materials folder"
                            return True
            
            # Check breadcrumb or current path in HTML
            all_steps = self.find_step_html("step_*")
            for step_file in all_steps:
                if self.search_in_html(step_file, "Project Materials") and \
                   (self.search_in_html(step_file, "breadcrumb") or \
                    self.search_in_html(step_file, "current-folder")):
                    self.details['cp2_navigated_to_folder'] = "Found Project Materials in navigation"
                    return True
            
            self.issues.append("Agent never navigated into Project Materials folder")
            self.details['cp2_navigated_to_folder'] = "No navigation to folder detected"
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking navigation: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Document.pdf was uploaded to Project Materials folder"""
        try:
            # Check if upload action was performed
            trajectory = self.get_trajectory()
            
            upload_attempted = False
            for entry in trajectory:
                if 'action' in entry and entry['action'].get('action_type') == 'UPLOAD':
                    upload_attempted = True
                    params = entry['action'].get('parameters', {})
                    if 'Document.pdf' in str(params):
                        self.details['cp3_document_uploaded'] = "Document.pdf upload attempted"
                        # Check if upload was successful in result
                        if entry.get('result', {}).get('success'):
                            return True
            
            # Check in HTML for the file
            final_html = self.find_final_html()
            if final_html and self.search_in_html(final_html, "Document.pdf"):
                self.details['cp3_document_uploaded'] = "Document.pdf found in final state"
                return True
            
            # Check intermediate steps
            if self.check_folder_in_intermediate_steps("Document.pdf"):
                self.details['cp3_document_uploaded'] = "Document.pdf found in intermediate step"
                return True
            
            if upload_attempted:
                self.issues.append("Document.pdf upload was attempted but may have failed")
            else:
                self.issues.append("Document.pdf was not uploaded")
            
            self.details['cp3_document_uploaded'] = "File not found"
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Document.pdf upload: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Design.png was uploaded to Project Materials folder"""
        try:
            # Check if upload action was performed
            trajectory = self.get_trajectory()
            
            upload_attempted = False
            for entry in trajectory:
                if 'action' in entry and entry['action'].get('action_type') == 'UPLOAD':
                    upload_attempted = True
                    params = entry['action'].get('parameters', {})
                    if 'Design.png' in str(params):
                        self.details['cp4_design_uploaded'] = "Design.png upload attempted"
                        # Check if upload was successful in result
                        if entry.get('result', {}).get('success'):
                            return True
            
            # Check in HTML for the file
            final_html = self.find_final_html()
            if final_html and self.search_in_html(final_html, "Design.png"):
                self.details['cp4_design_uploaded'] = "Design.png found in final state"
                return True
            
            # Check intermediate steps
            if self.check_folder_in_intermediate_steps("Design.png"):
                self.details['cp4_design_uploaded'] = "Design.png found in intermediate step"
                return True
            
            if upload_attempted:
                self.issues.append("Design.png upload was attempted but may have failed")
            else:
                self.issues.append("Design.png was not uploaded")
            
            self.details['cp4_design_uploaded'] = "File not found"
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking Design.png upload: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_folder_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_navigated_to_folder'] = self.check_checkpoint_2()
            self.checkpoints['cp3_document_uploaded'] = self.check_checkpoint_3()
            self.checkpoints['cp4_design_uploaded'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 2,
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
                'query_id': 2,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python file_query2.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Evaluator for note Query 5

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
        self.in_text = False
        self.current_tag = None
        self.text_content = []
        self.folder_names = []
        self.document_titles = []
        self.document_content = []
        self.in_folder_section = False
        self.in_document_section = False
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        
    def handle_data(self, data):
        text = data.strip()
        if text:
            self.text_content.append(text)
            # Look for folder names
            if 'Sprint Documents' in text:
                self.folder_names.append(text)
            # Look for document titles
            if 'Sprint 12 Retrospective' in text:
                self.document_titles.append(text)
            # Look for document content
            if 'What went well' in text or 'What to improve' in text or 'Action items' in text:
                self.document_content.append(text)


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_navigated_to_personal_workspace": False,
            "cp2_folder_created": False,
            "cp3_document_created_with_title": False,
            "cp4_document_content_added": False,
            "cp5_document_saved_in_folder": False,
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

    def parse_html_content(self, html_path: Path) -> TaskHTMLParser:
        """Parse HTML file and extract relevant information."""
        parser = TaskHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            parser.feed(f.read())
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigate to Personal Workspace"""
        try:
            trajectory = self.get_trajectory()
            if not trajectory:
                self.issues.append("No trajectory data found")
                return False
            
            # Check if agent clicked on Personal Workspace
            for entry in trajectory:
                action = entry.get('action', {})
                reasoning = action.get('reasoning', '').lower()
                
                if 'personal workspace' in reasoning:
                    self.details['navigated_to_personal_workspace'] = True
                    return True
            
            # Also check if any step HTML shows Personal Workspace context
            step_files = self.find_step_html("step_[3-9]")
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Personal Workspace' in content and 'New Folder' in content:
                        self.details['navigated_to_personal_workspace'] = True
                        return True
            
            self.issues.append("Did not navigate to Personal Workspace")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Folder 'Sprint Documents' created"""
        try:
            # Check intermediate steps (after folder creation around step 6-8)
            step_files = self.find_step_html("step_[6-9]")
            
            for step_file in step_files:
                parser = self.parse_html_content(step_file)
                if any('Sprint Documents' in name for name in parser.folder_names):
                    self.details['sprint_documents_folder_created'] = True
                    return True
                
                # Also check raw HTML
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Sprint Documents' in content:
                        self.details['sprint_documents_folder_created'] = True
                        return True
            
            self.issues.append("Sprint Documents folder not created")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Document created with title 'Sprint 12 Retrospective'"""
        try:
            # Check all steps after document creation attempt (step 8 onwards)
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Sprint 12 Retrospective' in content:
                        self.details['document_title_set'] = True
                        return True
            
            # Check final HTML
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Sprint 12 Retrospective' in content:
                        self.details['document_title_set'] = True
                        return True
            
            self.issues.append("Document title 'Sprint 12 Retrospective' not set")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Document content added"""
        try:
            required_content_phrases = [
                "What went well",
                "Team collaboration improved significantly",
                "What to improve",
                "Need better estimation for complex tasks",
                "Action items",
                "Implement story point poker for sprint planning"
            ]
            
            # Check all steps
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    matches = sum(1 for phrase in required_content_phrases if phrase in content)
                    if matches >= 3:  # At least half of the key phrases
                        self.details['document_content_added'] = True
                        self.details['content_phrases_found'] = matches
                        return True
            
            # Check final HTML
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    content = f.read()
                    matches = sum(1 for phrase in required_content_phrases if phrase in content)
                    if matches >= 3:
                        self.details['document_content_added'] = True
                        self.details['content_phrases_found'] = matches
                        return True
            
            self.issues.append("Document content not added or incomplete")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Document saved in Sprint Documents folder"""
        try:
            # Check if the document is associated with Sprint Documents folder
            # This would be visible in the document editor's folder selector
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for both the document title and Sprint Documents folder in same context
                    if 'Sprint 12 Retrospective' in content and 'Sprint Documents' in content:
                        # Additional check: look for folder selection indicators
                        if 'selected' in content or 'value' in content:
                            self.details['document_saved_in_folder'] = True
                            return True
            
            # Check trajectory for successful save action with folder
            trajectory = self.get_trajectory()
            for entry in trajectory:
                action = entry.get('action', {})
                reasoning = action.get('reasoning', '').lower()
                if 'save' in reasoning and 'sprint documents' in reasoning:
                    self.details['document_saved_in_folder'] = True
                    return True
            
            self.issues.append("Document not saved in Sprint Documents folder")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 5: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_personal_workspace'] = self.check_checkpoint_1()
            self.checkpoints['cp2_folder_created'] = self.check_checkpoint_2()
            self.checkpoints['cp3_document_created_with_title'] = self.check_checkpoint_3()
            self.checkpoints['cp4_document_content_added'] = self.check_checkpoint_4()
            self.checkpoints['cp5_document_saved_in_folder'] = self.check_checkpoint_5()

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
        print("Usage: python note_query5.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

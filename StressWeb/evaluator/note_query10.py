#!/usr/bin/env python3
# Evaluator for note Query 10

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class TaskHTMLParser(HTMLParser):
    """Parse HTML to extract task-specific information."""

    def __init__(self):
        super().__init__()
        self.in_script = False
        self.script_content = ""
        self.document_data = None

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False
            # Try to parse script content for document data
            if "Research Summary Q1 2026" in self.script_content or "Key Findings" in self.script_content:
                self.document_data = self.script_content
            self.script_content = ""

    def handle_data(self, data):
        if self.in_script:
            self.script_content += data


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_document_created": False,
            "cp2_title_set": False,
            "cp3_heading_added": False,
            "cp4_paragraph_added": False,
            "cp5_tags_added": False,
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
        """Checkpoint 1: Document was created"""
        html_file = self.find_final_html()
        if not html_file:
            self.issues.append("No final HTML file found")
            return False

        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if document was created (not just on dashboard)
            if "Blank Note" in content and "Create New" in content:
                self.issues.append("Agent is still on dashboard, document not created")
                return False

            # Check if editor is visible
            if "ProseMirror" in content or "editor" in content.lower():
                self.details["document_created"] = True
                return True

            self.issues.append("Document editor not found in final state")
            return False

        except Exception as e:
            self.issues.append(f"Error checking document creation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Document title is 'Research Summary Q1 2026'"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for title in various possible locations
            title_found = "Research Summary Q1 2026" in content
            
            if title_found:
                self.details["title"] = "Research Summary Q1 2026"
                return True
            
            # Check if still Untitled Document
            if "Untitled Document" in content:
                self.issues.append("Document title is still 'Untitled Document'")
                self.details["title"] = "Untitled Document"
            else:
                self.issues.append("Document title 'Research Summary Q1 2026' not found")
                self.details["title"] = "Unknown"

            return False

        except Exception as e:
            self.issues.append(f"Error checking title: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Heading 'Key Findings' was added"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for "Key Findings" heading
            if "Key Findings" in content:
                self.details["heading"] = "Key Findings"
                return True

            self.issues.append("Heading 'Key Findings' not found in document")
            return False

        except Exception as e:
            self.issues.append(f"Error checking heading: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Required paragraph was added"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for key phrases from the required paragraph
            required_text1 = "Our Q1 research indicates strong market demand for AI-powered productivity tools"
            required_text2 = "78% of respondents are willing to adopt AI assistants"

            has_paragraph = required_text1 in content or "Our Q1 research" in content

            if has_paragraph:
                self.details["paragraph_added"] = True
                return True

            self.issues.append("Required paragraph content not found in document")
            return False

        except Exception as e:
            self.issues.append(f"Error checking paragraph: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Tags 'research', 'quarterly', and 'analysis' were added"""
        html_file = self.find_final_html()
        if not html_file:
            return False

        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for tags in the document
            required_tags = ["research", "quarterly", "analysis"]
            found_tags = []

            for tag in required_tags:
                # Look for tag in various formats
                tag_escaped = re.escape(tag)
                if re.search(rf'["\']tag["\'][^}}]*{tag_escaped}', content, re.IGNORECASE) or \
                   re.search(rf'{tag_escaped}["\'][^}}]*tag', content, re.IGNORECASE) or \
                   re.search(rf'<span[^>]*class="[^"]*tag[^"]*"[^>]*>{tag_escaped}</span>', content, re.IGNORECASE):
                    found_tags.append(tag)

            self.details["tags_found"] = found_tags
            self.details["tags_required"] = required_tags

            if len(found_tags) == len(required_tags):
                return True

            missing_tags = set(required_tags) - set(found_tags)
            self.issues.append(f"Missing tags: {', '.join(missing_tags)}")
            return False

        except Exception as e:
            self.issues.append(f"Error checking tags: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_document_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_title_set'] = self.check_checkpoint_2()
            self.checkpoints['cp3_heading_added'] = self.check_checkpoint_3()
            self.checkpoints['cp4_paragraph_added'] = self.check_checkpoint_4()
            self.checkpoints['cp5_tags_added'] = self.check_checkpoint_5()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 10,
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
                'query_id': 10,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python note_query10.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

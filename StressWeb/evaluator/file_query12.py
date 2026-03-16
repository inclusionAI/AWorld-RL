#!/usr/bin/env python3
# Evaluator for file Query 12

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
        self.in_file_name = False
        self.in_folder_name = False
        self.in_result_item = False
        self.in_search_query = False
        self.in_filter_badge = False
        self.in_selected_count = False
        self.in_trash_empty = False
        self.in_trash_item = False
        self.current_tag = None
        self.current_attrs = {}
        
        self.folders = []
        self.files = []
        self.search_results = []
        self.search_query = ""
        self.filter_badge = ""
        self.selected_count = ""
        self.trash_empty = False
        self.trash_items = []
        self.current_result = {}
        self.current_trash_item = {}

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        if tag == "h3":
            class_attr = self.current_attrs.get("class", "")
            if "file-name" in class_attr or "folder-name" in class_attr:
                if "folder-name" in class_attr:
                    self.in_folder_name = True
                else:
                    self.in_file_name = True
        
        if tag == "div":
            class_attr = self.current_attrs.get("class", "")
            if "result-item" in class_attr:
                self.in_result_item = True
                self.current_result = {"selected": "selected" in class_attr}
            if "trash-item" in class_attr:
                self.in_trash_item = True
                self.current_trash_item = {}
        
        if tag == "p":
            class_attr = self.current_attrs.get("class", "")
            if "search-query" in class_attr:
                self.in_search_query = True
                
        if tag == "span":
            class_attr = self.current_attrs.get("class", "")
            if "filter-badge" in class_attr:
                self.in_filter_badge = True
            if "selected-count" in class_attr:
                self.in_selected_count = True
                
        if tag == "h2":
            # Check for "Trash is empty" text in next data
            pass

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_folder_name:
            self.folders.append(data)
            self.in_folder_name = False
            
        if self.in_file_name:
            if self.in_result_item:
                self.current_result["name"] = data
            elif self.in_trash_item:
                self.current_trash_item["name"] = data
            else:
                self.files.append(data)
            self.in_file_name = False
            
        if self.in_search_query and "Showing results for:" in data:
            # Extract search term from text like 'Showing results for: "welcome"'
            match = re.search(r'"([^"]+)"', data)
            if match:
                self.search_query = match.group(1)
            self.in_search_query = False
            
        if self.in_filter_badge:
            self.filter_badge = data
            self.in_filter_badge = False
            
        if self.in_selected_count:
            self.selected_count = data
            self.in_selected_count = False
            
        if "Trash is empty" in data:
            self.trash_empty = True

    def handle_endtag(self, tag):
        if tag == "div":
            if self.in_result_item:
                if self.current_result.get("name"):
                    self.search_results.append(self.current_result)
                self.current_result = {}
                self.in_result_item = False
            if self.in_trash_item:
                if self.current_trash_item.get("name"):
                    self.trash_items.append(self.current_trash_item)
                self.current_trash_item = {}
                self.in_trash_item = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_folder_created": False,
            "cp2_files_uploaded": False,
            "cp3_welcome_guide_deleted": False,
            "cp4_file_restored_from_trash": False,
            "cp5_search_executed": False,
            "cp6_filtered_files_deleted": False,
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

    def parse_html(self, html_path: Path) -> TaskHTMLParser:
        """Parse HTML file and return parser with extracted data."""
        parser = TaskHTMLParser()
        with open(html_path, 'r', encoding='utf-8') as f:
            parser.feed(f.read())
        return parser

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Verify 'Project Materials' folder was created"""
        try:
            # Check in trajectory if folder was created
            trajectory = self.get_trajectory()
            for entry in trajectory:
                action = entry.get("action", {})
                if action.get("action_type") == "FILL":
                    text = action.get("parameters", {}).get("text", "")
                    if text == "Project Materials":
                        self.details["folder_creation_step"] = entry.get("step")
                        return True
            
            self.issues.append("Project Materials folder was not created")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking folder creation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Verify Document.pdf and Design.png were uploaded to Project Materials"""
        try:
            # Check all HTML files for uploaded files in Project Materials folder
            all_steps = list(self.result_dir.glob("step_*_raw.html"))
            final_html = self.find_final_html()
            if final_html:
                all_steps.append(final_html)
            
            found_document = False
            found_design = False
            
            for html_file in all_steps:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "Document.pdf" in content and "Project Materials" in content:
                        found_document = True
                    if "Design.png" in content and "Project Materials" in content:
                        found_design = True
                        
            if not found_document and not found_design:
                self.issues.append("Neither Document.pdf nor Design.png were uploaded to Project Materials")
            elif not found_document:
                self.issues.append("Document.pdf was not uploaded to Project Materials")
            elif not found_design:
                self.issues.append("Design.png was not uploaded to Project Materials")
            else:
                self.details["files_uploaded"] = ["Document.pdf", "Design.png"]
                return True
                
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking file uploads: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Verify welcome_guide.pdf was deleted"""
        try:
            # Check trajectory for delete action
            trajectory = self.get_trajectory()
            deleted = False
            
            for entry in trajectory:
                action = entry.get("action", {})
                reasoning = action.get("reasoning", "")
                
                # Look for delete action on welcome_guide.pdf
                if "welcome_guide.pdf" in reasoning and ("delete" in reasoning.lower() or "Delete" in reasoning):
                    deleted = True
                    self.details["welcome_guide_delete_step"] = entry.get("step")
                    break
            
            # Also check if file appears in trash in any step
            trash_steps = self.find_step_html("step_3[89]*")
            trash_steps.extend(self.find_step_html("step_4[0-2]*"))
            
            for trash_file in trash_steps:
                with open(trash_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "welcome_guide.pdf" in content and "trash" in content.lower():
                        deleted = True
                        break
            
            if not deleted:
                self.issues.append("welcome_guide.pdf was not deleted")
                return False
                
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking welcome_guide.pdf deletion: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Verify file was restored from trash"""
        try:
            # Check trash pages to see if file was restored
            trash_steps = self.find_step_html("step_3[89]*")
            trash_steps.extend(self.find_step_html("step_4[0-3]*"))
            
            # Parse trash HTML to check if it was empty or had items
            found_trash_with_file = False
            found_restore_action = False
            
            for trash_file in trash_steps:
                parser = self.parse_html(trash_file)
                if parser.trash_items:
                    found_trash_with_file = True
                    self.details["trash_had_items"] = True
                    
            # Check trajectory for restore action
            trajectory = self.get_trajectory()
            for entry in trajectory:
                action = entry.get("action", {})
                reasoning = action.get("reasoning", "")
                
                if "restore" in reasoning.lower():
                    found_restore_action = True
                    self.details["restore_step"] = entry.get("step")
                    break
            
            # According to task, the trash was empty, so restoration couldn't happen
            if not found_trash_with_file:
                self.issues.append("Trash was empty - welcome_guide.pdf was not in trash, so could not be restored")
                self.details["trash_empty"] = True
                return False
                
            if not found_restore_action:
                self.issues.append("No restore action was performed")
                return False
                
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking file restoration: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Verify search for 'welcome' was executed"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
                
            parser = self.parse_html(final_html)
            
            if parser.search_query == "welcome":
                self.details["search_query"] = "welcome"
                self.details["search_results_count"] = len(parser.search_results)
                return True
            else:
                self.issues.append(f"Search was not executed for 'welcome' (found: '{parser.search_query}')")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking search execution: {str(e)}")
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Verify files between 1MB-10MB were deleted"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
                
            parser = self.parse_html(final_html)
            
            # Check if filter was applied
            if "1-10 MB" in parser.filter_badge or "1MB - 10MB" in parser.filter_badge:
                self.details["filter_applied"] = parser.filter_badge
            else:
                self.issues.append("File size filter (1-10 MB) was not applied")
                return False
            
            # Check if files were selected for deletion
            selected_files = [r for r in parser.search_results if r.get("selected")]
            
            if not selected_files:
                self.issues.append("No files were selected for deletion")
                return False
                
            self.details["selected_files_count"] = len(selected_files)
            self.details["selected_files"] = [f.get("name") for f in selected_files]
            
            # Check trajectory for delete action
            trajectory = self.get_trajectory()
            delete_action_found = False
            
            for entry in trajectory:
                action = entry.get("action", {})
                if action.get("action_type") == "CLICK":
                    selector = action.get("parameters", {}).get("selector", "")
                    reasoning = action.get("reasoning", "")
                    if "Delete" in selector or ("delete" in reasoning.lower() and "selected" in reasoning.lower()):
                        delete_action_found = True
                        self.details["delete_action_step"] = entry.get("step")
                        break
            
            if not delete_action_found:
                self.issues.append("Delete action was not performed on selected files")
                return False
                
            return True
            
        except Exception as e:
            self.issues.append(f"Error checking file deletion: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_folder_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_files_uploaded'] = self.check_checkpoint_2()
            self.checkpoints['cp3_welcome_guide_deleted'] = self.check_checkpoint_3()
            self.checkpoints['cp4_file_restored_from_trash'] = self.check_checkpoint_4()
            self.checkpoints['cp5_search_executed'] = self.check_checkpoint_5()
            self.checkpoints['cp6_filtered_files_deleted'] = self.check_checkpoint_6()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 12,
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
                'query_id': 12,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python file_query12.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

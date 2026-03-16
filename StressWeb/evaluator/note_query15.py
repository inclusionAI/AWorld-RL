#!/usr/bin/env python3
# Evaluator for note Query 15

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
        self.in_editor = False
        self.in_title = False
        self.in_breadcrumb = False
        self.in_select = False
        self.in_tags = False
        self.current_tag = None
        self.current_attrs = {}
        
        self.title = ""
        self.breadcrumb = ""
        self.editor_content = ""
        self.workspace_selected = None
        self.folder_selected = None
        self.tags = []
        self.workspace_options = []
        self.folder_options = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        self.current_attrs = attrs_dict
        
        if tag == "input" and attrs_dict.get("class") == "title-input":
            self.in_title = True
            self.title = attrs_dict.get("value", "")
            
        if tag == "span" and attrs_dict.get("class") == "breadcrumb":
            self.in_breadcrumb = True
            
        if tag == "div" and "editor-content" in attrs_dict.get("class", ""):
            self.in_editor = True
            
        if tag == "select" and "sidebar-select" in attrs_dict.get("class", ""):
            self.in_select = True
            
        if tag == "div" and "tags-container" in attrs_dict.get("class", ""):
            self.in_tags = True
            
    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
            
        if self.in_breadcrumb:
            self.breadcrumb = data
            
        if self.in_editor:
            self.editor_content += data + " "
            
        if self.in_select and self.current_tag == "option":
            option_value = self.current_attrs.get("value", "")
            if "Workspace" in self.breadcrumb or "workspace" in str(self.current_attrs):
                self.workspace_options.append((option_value, data))
                if self.current_attrs.get("selected"):
                    self.workspace_selected = data
            else:
                self.folder_options.append((option_value, data))
                if self.current_attrs.get("selected"):
                    self.folder_selected = data
                    
        if self.in_tags and self.current_tag == "span" and "tag-item" in self.current_attrs.get("class", ""):
            self.tags.append(data)
            
    def handle_endtag(self, tag):
        if tag == "div" and self.in_editor:
            self.in_editor = False
        if tag == "span" and self.in_breadcrumb:
            self.in_breadcrumb = False
        if tag == "select" and self.in_select:
            self.in_select = False
        if tag == "div" and self.in_tags:
            self.in_tags = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        self.checkpoints = {
            "cp1_workspace_created": False,
            "cp2_folder_created": False,
            "cp3_document_structure": False,
            "cp4_tags_added": False,
            "cp5_saved_no_workspace": False,
            "cp6_shareable_link": False,
        }
        
        self.issues = []
        self.details = {}
    
    def find_final_html(self) -> Optional[Path]:
        """Find final HTML state file."""
        final_files = list(self.result_dir.glob("final_*_raw.html"))
        if final_files:
            return final_files[0]
        step_files = sorted(self.result_dir.glob("step_*_raw.html"))
        return step_files[-1] if step_files else None
    
    def find_step_html(self, step_pattern: str) -> List[Path]:
        """Find intermediate step HTML files."""
        return sorted(self.result_dir.glob(f"{step_pattern}_raw.html"))
    
    def get_trajectory(self) -> List[Dict]:
        """Load trajectory data."""
        if not self.traj_file.exists():
            return []
        trajectory = []
        with open(self.traj_file, 'r', encoding='utf-8') as f:
            for line in f:
                trajectory.append(json.loads(line))
        return trajectory
    
    def load_result_json(self) -> Dict:
        """Load result.json."""
        with open(self.result_file, 'r') as f:
            return json.load(f)
    
    def parse_html_file(self, html_file: Path) -> TaskHTMLParser:
        """Parse HTML file and extract information."""
        parser = TaskHTMLParser()
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            parser.feed(content)
        return parser
    
    def check_checkpoint_1(self) -> bool:
        """Check if workspace 'Project Alpha' was created."""
        trajectory = self.get_trajectory()
        
        for action in trajectory:
            if action.get("action", {}).get("action_type") == "FILL":
                text = action.get("action", {}).get("parameters", {}).get("text", "")
                if "Project Alpha" in text:
                    self.details["workspace_name_entered"] = True
                    
        final_html = self.find_final_html()
        if final_html:
            parser = self.parse_html_file(final_html)
            for value, name in parser.workspace_options:
                if "Project Alpha" in name:
                    self.details["workspace_found"] = True
                    return True
                    
        self.issues.append("Workspace 'Project Alpha' not found")
        return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if folder 'Planning Documents' was created."""
        trajectory = self.get_trajectory()
        
        for action in trajectory:
            if action.get("action", {}).get("action_type") == "FILL":
                text = action.get("action", {}).get("parameters", {}).get("text", "")
                if "Planning Documents" in text:
                    self.details["folder_name_entered"] = True
                    return True
                    
        all_html = self.find_step_html("step_*")
        for html_file in all_html:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "Planning Documents" in content or "Planning Doc" in content:
                    self.details["folder_found_in_html"] = True
                    return True
                    
        self.issues.append("Folder 'Planning Documents' not found")
        return False
    
    def check_checkpoint_3(self) -> bool:
        """Check document structure: title, H2 headings, and paragraphs."""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML found")
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parser = self.parse_html_file(final_html)
        
        has_title = "Project Charter" in parser.title or "Project Charter" in content
        has_project_overview = "Project Overview" in content
        has_objectives = "Objectives" in content
        has_para1 = "Project Alpha aims to deliver a next-generation analytics platform by Q3 2026" in content
        has_para2 = "Increase user engagement by 40%" in content and "reduce data processing time by 60%" in content and "99.9% uptime" in content
        
        self.details["title_check"] = has_title
        self.details["project_overview_heading"] = has_project_overview
        self.details["objectives_heading"] = has_objectives
        self.details["paragraph_1"] = has_para1
        self.details["paragraph_2"] = has_para2
        
        if has_title and has_project_overview and has_objectives and has_para1 and has_para2:
            return True
            
        if not has_title:
            self.issues.append("Document title 'Project Charter' not found")
        if not has_project_overview:
            self.issues.append("H2 heading 'Project Overview' not found")
        if not has_objectives:
            self.issues.append("H2 heading 'Objectives' not found")
        if not has_para1:
            self.issues.append("First paragraph content incomplete")
        if not has_para2:
            self.issues.append("Second paragraph content incomplete")
            
        return False
    
    def check_checkpoint_4(self) -> bool:
        """Check if tags 'project', 'planning', and 'alpha' were added."""
        final_html = self.find_final_html()
        if not final_html:
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        trajectory = self.get_trajectory()
        tags_entered = []
        
        for action in trajectory:
            if action.get("action", {}).get("action_type") == "FILL":
                selector = action.get("action", {}).get("parameters", {}).get("selector", "")
                text = action.get("action", {}).get("parameters", {}).get("text", "")
                if "tag" in selector.lower() and text.lower() in ["project", "planning", "alpha"]:
                    tags_entered.append(text.lower())
                    
        has_project = "project" in tags_entered or 'tag-input' in content
        has_planning = "planning" in tags_entered
        has_alpha = "alpha" in tags_entered
        
        self.details["tags_entered"] = tags_entered
        self.details["expected_tags"] = ["project", "planning", "alpha"]
        
        if len(tags_entered) >= 3 and has_project and has_planning and has_alpha:
            return True
            
        self.issues.append(f"Tags incomplete. Found: {tags_entered}, Expected: ['project', 'planning', 'alpha']")
        return False
    
    def check_checkpoint_5(self) -> bool:
        """Check if document was saved without workspace association."""
        final_html = self.find_final_html()
        if not final_html:
            return False
            
        parser = self.parse_html_file(final_html)
        
        trajectory = self.get_trajectory()
        save_clicked = False
        
        for action in trajectory:
            if action.get("action", {}).get("action_type") == "CLICK":
                selector = action.get("action", {}).get("parameters", {}).get("selector", "")
                if "save" in selector.lower() or "Save Document" in str(action.get("action", {}).get("parameters")):
                    save_clicked = True
                    break
                    
        self.details["save_button_clicked"] = save_clicked
        self.details["workspace_selected"] = parser.workspace_selected
        
        if save_clicked:
            return True
            
        self.issues.append("Save button not clicked or document not saved")
        return False
    
    def check_checkpoint_6(self) -> bool:
        """Check if shareable link was generated."""
        trajectory = self.get_trajectory()
        
        for action in trajectory:
            action_type = action.get("action", {}).get("action_type", "")
            selector = action.get("action", {}).get("parameters", {}).get("selector", "")
            reasoning = action.get("action", {}).get("reasoning", "")
            
            if "share" in selector.lower() or "share" in reasoning.lower():
                self.details["share_action_found"] = True
                
            if "view-only" in reasoning.lower() or "view only" in reasoning.lower():
                self.details["view_only_mentioned"] = True
                
            if "30" in reasoning.lower() and "day" in reasoning.lower():
                self.details["30_day_expiration"] = True
                
        all_html = self.find_step_html("step_*")
        for html_file in all_html[-20:]:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "share" in content.lower() and ("link" in content.lower() or "url" in content.lower()):
                    self.details["share_ui_found"] = True
                    
        if self.details.get("share_action_found") or self.details.get("share_ui_found"):
            return True
            
        self.issues.append("Shareable link generation not completed")
        return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            self.checkpoints['cp1_workspace_created'] = self.check_checkpoint_1()
            self.checkpoints['cp2_folder_created'] = self.check_checkpoint_2()
            self.checkpoints['cp3_document_structure'] = self.check_checkpoint_3()
            self.checkpoints['cp4_tags_added'] = self.check_checkpoint_4()
            self.checkpoints['cp5_saved_no_workspace'] = self.check_checkpoint_5()
            self.checkpoints['cp6_shareable_link'] = self.check_checkpoint_6()
            
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 15,
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
                'query_id': 15,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python note_query15.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

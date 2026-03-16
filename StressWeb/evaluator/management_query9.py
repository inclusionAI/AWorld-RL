#!/usr/bin/env python3
# Evaluator for management Query 9

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class TagHTMLParser(HTMLParser):
    """Parse HTML to extract tag information from the Settings page."""

    def __init__(self):
        super().__init__()
        self.tags = []
        self.current_tag = {}
        self.in_tag_item = False
        self.in_tag_name = False
        self.in_tag_color = False
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Detect tag-item div
        if tag == 'div' and 'class' in attrs_dict and 'tag-item' in attrs_dict.get('class', ''):
            self.in_tag_item = True
            self.current_tag = {}
            # Extract border color from style attribute
            style = attrs_dict.get('style', '')
            color_match = re.search(r'border-left:\s*4px\s+solid\s+rgb\((\d+),\s*(\d+),\s*(\d+)\)', style)
            if color_match:
                r, g, b = map(int, color_match.groups())
                self.current_tag['color_rgb'] = (r, g, b)
        
        # Detect tag-name span
        elif self.in_tag_item and tag == 'span' and 'class' in attrs_dict and 'tag-name' in attrs_dict.get('class', ''):
            self.in_tag_name = True
            self.current_data = []
        
        # Detect tag-color span
        elif self.in_tag_item and tag == 'span' and 'class' in attrs_dict and 'tag-color' in attrs_dict.get('class', ''):
            self.in_tag_color = True
            # Extract background color from style attribute
            style = attrs_dict.get('style', '')
            color_match = re.search(r'background-color:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\)', style)
            if color_match:
                r, g, b = map(int, color_match.groups())
                self.current_tag['bg_color_rgb'] = (r, g, b)

    def handle_data(self, data):
        if self.in_tag_name:
            self.current_data.append(data)

    def handle_endtag(self, tag):
        if tag == 'span' and self.in_tag_name:
            self.current_tag['name'] = ''.join(self.current_data).strip()
            self.in_tag_name = False
            self.current_data = []
        elif tag == 'span' and self.in_tag_color:
            self.in_tag_color = False
        elif tag == 'div' and self.in_tag_item:
            if self.current_tag:
                self.tags.append(self.current_tag)
            self.current_tag = {}
            self.in_tag_item = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_settings": False,
            "cp2_tag_created": False,
            "cp3_tag_name_correct": False,
            "cp4_tag_color_correct": False,
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
        """Checkpoint 1: Navigated to Settings page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if Settings page was visited in trajectory
            for step in trajectory:
                url = step.get('url', '')
                if '/settings' in url.lower():
                    self.details['navigated_to_settings'] = True
                    return True
            
            # Also check final HTML
            final_html = self.find_final_html()
            if final_html:
                with open(final_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    if 'settings' in html_content.lower() and 'Settings' in html_content:
                        self.details['navigated_to_settings'] = True
                        return True
            
            self.issues.append("Agent did not navigate to the Settings page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking navigation to Settings: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Tag was created (Important Meetings exists in final state)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TagHTMLParser()
            parser.feed(html_content)
            
            self.details['total_tags'] = len(parser.tags)
            self.details['tags_found'] = [tag.get('name', '') for tag in parser.tags]
            
            # Check if "Important Meetings" tag exists
            for tag in parser.tags:
                if tag.get('name', '').strip() == "Important Meetings":
                    self.details['important_meetings_tag'] = tag
                    return True
            
            self.issues.append("Tag 'Important Meetings' was not created or not found in final state")
            return False
        except Exception as e:
            self.issues.append(f"Error checking tag creation: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Tag name is exactly 'Important Meetings'"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TagHTMLParser()
            parser.feed(html_content)
            
            # Find the "Important Meetings" tag
            for tag in parser.tags:
                tag_name = tag.get('name', '').strip()
                if tag_name == "Important Meetings":
                    self.details['tag_name_verified'] = tag_name
                    return True
            
            self.issues.append("Tag name 'Important Meetings' not found with exact match")
            return False
        except Exception as e:
            self.issues.append(f"Error checking tag name: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Tag color is RGB(75, 130, 130)"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()

            parser = TagHTMLParser()
            parser.feed(html_content)
            
            # Find the "Important Meetings" tag and check its color
            for tag in parser.tags:
                if tag.get('name', '').strip() == "Important Meetings":
                    # Check border-left color (this is the main color indicator)
                    border_color = tag.get('color_rgb')
                    bg_color = tag.get('bg_color_rgb')
                    
                    self.details['tag_border_color'] = border_color
                    self.details['tag_bg_color'] = bg_color
                    
                    # The expected color is RGB(75, 130, 130)
                    expected_color = (75, 130, 130)
                    
                    # Check if either border or background color matches
                    if border_color == expected_color or bg_color == expected_color:
                        self.details['tag_color_verified'] = True
                        return True
                    
                    # Allow small tolerance (within 5 units per channel)
                    if border_color:
                        diff = sum(abs(a - b) for a, b in zip(border_color, expected_color))
                        if diff <= 15:  # Total tolerance of 15 across all channels
                            self.details['tag_color_verified'] = True
                            self.details['color_tolerance_used'] = diff
                            return True
                    
                    if bg_color:
                        diff = sum(abs(a - b) for a, b in zip(bg_color, expected_color))
                        if diff <= 15:
                            self.details['tag_color_verified'] = True
                            self.details['color_tolerance_used'] = diff
                            return True
                    
                    self.issues.append(f"Tag color mismatch. Expected RGB(75, 130, 130), found border={border_color}, bg={bg_color}")
                    return False
            
            self.issues.append("Tag 'Important Meetings' not found to check color")
            return False
        except Exception as e:
            self.issues.append(f"Error checking tag color: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_settings'] = self.check_checkpoint_1()
            self.checkpoints['cp2_tag_created'] = self.check_checkpoint_2()
            self.checkpoints['cp3_tag_name_correct'] = self.check_checkpoint_3()
            self.checkpoints['cp4_tag_color_correct'] = self.check_checkpoint_4()

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
        print("Usage: python management_query9.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

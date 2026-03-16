#!/usr/bin/env python3
# Evaluator for calendar Query 11

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class EventFormParser(HTMLParser):
    """Parse HTML to extract event form information."""

    def __init__(self):
        super().__init__()
        self.in_modal = False
        self.in_form = False
        self.current_input_name = None
        self.current_select_name = None
        self.current_textarea_name = None
        self.in_textarea = False
        self.form_data = {}
        self.color_selected = None
        self.in_color_picker = False
        self.current_color = None
        self.current_tag = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict:
            if 'modal-overlay' in attrs_dict['class']:
                self.in_modal = True
            elif 'color-picker' in attrs_dict['class']:
                self.in_color_picker = True
                
        if tag == 'form' and self.in_modal:
            self.in_form = True
            
        if self.in_form and tag == 'input':
            name = attrs_dict.get('name', '')
            value = attrs_dict.get('value', '')
            if name:
                self.form_data[name] = value
                
        if self.in_form and tag == 'select':
            self.current_select_name = attrs_dict.get('name', '')
            
        if self.in_form and tag == 'option':
            if self.current_select_name and 'selected' in attrs_dict:
                value = attrs_dict.get('value', '')
                self.form_data[self.current_select_name] = value
                
        if self.in_form and tag == 'textarea':
            self.in_textarea = True
            self.current_textarea_name = attrs_dict.get('name', '')
            
        if self.in_color_picker and tag == 'button':
            if 'class' in attrs_dict and 'color-option' in attrs_dict['class']:
                style = attrs_dict.get('style', '')
                color_match = re.search(r'background-color:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\)', style)
                if color_match:
                    self.current_color = (int(color_match.group(1)), int(color_match.group(2)), int(color_match.group(3)))
                    if 'selected' in attrs_dict['class']:
                        self.color_selected = self.current_color
        
        self.current_tag = tag
        
    def handle_data(self, data):
        if self.in_textarea and self.current_textarea_name:
            self.form_data[self.current_textarea_name] = data.strip()
            
    def handle_endtag(self, tag):
        if tag == 'form':
            self.in_form = False
        if tag == 'textarea':
            self.in_textarea = False
            self.current_textarea_name = None
        if tag == 'select':
            self.current_select_name = None
        if tag == 'div':
            if self.in_color_picker:
                self.in_color_picker = False


class CalendarEventsParser(HTMLParser):
    """Parse HTML to extract calendar events from the month view."""
    
    def __init__(self):
        super().__init__()
        self.events = []
        self.in_day_cell = False
        self.current_day = None
        self.in_day_number = False
        self.in_event_title = False
        self.current_event = {}
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'div' and 'class' in attrs_dict:
            if 'day-cell' in attrs_dict['class'] and 'other-month' not in attrs_dict['class']:
                self.in_day_cell = True
            elif 'day-number' in attrs_dict['class'] and self.in_day_cell:
                self.in_day_number = True
            elif 'event-title' in attrs_dict['class'] and self.in_day_cell:
                self.in_event_title = True
            elif 'event-bar' in attrs_dict['class'] and self.in_day_cell:
                style = attrs_dict.get('style', '')
                color_match = re.search(r'background-color:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\)', style)
                if color_match:
                    self.current_event['color'] = (int(color_match.group(1)), int(color_match.group(2)), int(color_match.group(3)))
                    
    def handle_data(self, data):
        if self.in_day_number:
            self.current_day = data.strip()
        elif self.in_event_title and self.current_day:
            title = data.strip()
            if title:
                event = {
                    'day': self.current_day,
                    'title': title,
                    'color': self.current_event.get('color')
                }
                self.events.append(event)
                self.current_event = {}
                
    def handle_endtag(self, tag):
        if tag == 'div':
            if self.in_day_number:
                self.in_day_number = False
            elif self.in_event_title:
                self.in_event_title = False
            elif self.in_day_cell:
                self.in_day_cell = False
                self.current_day = None


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on the task requirements
        self.checkpoints = {
            "cp1_modal_opened": False,
            "cp2_title_filled": False,
            "cp3_date_set": False,
            "cp4_time_set": False,
            "cp5_duration_set": False,
            "cp6_location_set": False,
            "cp7_description_set": False,
            "cp8_category_set": False,
            "cp9_color_set": False,
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
        """Checkpoint 1: Modal was opened for event creation"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if modal is present and visible
        if 'modal-overlay' in content and 'Create New Event' in content:
            self.details['modal_opened'] = True
            return True
        else:
            self.issues.append("Event creation modal not found or not opened")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Event title is 'Q1 Planning Meeting'"""
        final_html = self.find_final_html()
        if not final_html:
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parser = EventFormParser()
        parser.feed(content)
        
        title = parser.form_data.get('title', '')
        self.details['title'] = title
        
        if title == 'Q1 Planning Meeting':
            return True
        else:
            self.issues.append(f"Event title incorrect: expected 'Q1 Planning Meeting', got '{title}'")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Date is set to January 30, 2026"""
        final_html = self.find_final_html()
        if not final_html:
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parser = EventFormParser()
        parser.feed(content)
        
        date = parser.form_data.get('start_date', '')
        self.details['date'] = date
        
        if date == '2026-01-30':
            return True
        else:
            self.issues.append(f"Date incorrect: expected '2026-01-30', got '{date}'")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Start time is set to 2:00 PM (14:00)"""
        final_html = self.find_final_html()
        if not final_html:
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parser = EventFormParser()
        parser.feed(content)
        
        time = parser.form_data.get('start_time', '')
        self.details['start_time'] = time
        
        if time == '14:00':
            return True
        else:
            self.issues.append(f"Start time incorrect: expected '14:00' (2:00 PM), got '{time}'")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Duration is set to 2 hours (120 minutes)"""
        final_html = self.find_final_html()
        if not final_html:
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for selected option in duration dropdown
        duration_match = re.search(r'<select name="duration"[^>]*>.*?<option value="120"[^>]*selected[^>]*>.*?</select>', content, re.DOTALL)
        if not duration_match:
            # Try alternative: check if value="120" is in select element
            duration_match = re.search(r'<select name="duration"[^>]*>.*?<option value="120"[^>]*>2 hours</option>.*?</select>', content, re.DOTALL)
        
        self.details['duration_check'] = bool(duration_match)
        
        if duration_match:
            return True
        else:
            self.issues.append("Duration not set to 2 hours (120 minutes)")
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Location is set to 'Conference Room A'"""
        final_html = self.find_final_html()
        if not final_html:
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parser = EventFormParser()
        parser.feed(content)
        
        location = parser.form_data.get('location', '')
        self.details['location'] = location
        
        if location == 'Conference Room A':
            return True
        else:
            self.issues.append(f"Location incorrect: expected 'Conference Room A', got '{location}'")
            return False

    def check_checkpoint_7(self) -> bool:
        """Checkpoint 7: Description is set to 'Discuss Q1 goals and budget allocation'"""
        final_html = self.find_final_html()
        if not final_html:
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parser = EventFormParser()
        parser.feed(content)
        
        description = parser.form_data.get('description', '')
        self.details['description'] = description
        
        if description == 'Discuss Q1 goals and budget allocation':
            return True
        else:
            self.issues.append(f"Description incorrect: expected 'Discuss Q1 goals and budget allocation', got '{description}'")
            return False

    def check_checkpoint_8(self) -> bool:
        """Checkpoint 8: Category is set to 'Work'"""
        final_html = self.find_final_html()
        if not final_html:
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for selected Work category (value="101" based on HTML structure)
        category_match = re.search(r'<select name="category_id"[^>]*>.*?<option value="101"[^>]*selected[^>]*>Work</option>.*?</select>', content, re.DOTALL)
        
        self.details['category_work_selected'] = bool(category_match)
        
        if category_match:
            return True
        else:
            self.issues.append("Category not set to 'Work'")
            return False

    def check_checkpoint_9(self) -> bool:
        """Checkpoint 9: Color is set to red"""
        final_html = self.find_final_html()
        if not final_html:
            return False
            
        with open(final_html, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parser = EventFormParser()
        parser.feed(content)
        
        # Red color in RGB: rgb(239, 68, 68)
        expected_red = (239, 68, 68)
        selected_color = parser.color_selected
        
        self.details['selected_color'] = selected_color
        
        if selected_color == expected_red:
            return True
        else:
            self.issues.append(f"Color not set to red: expected {expected_red}, got {selected_color}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_modal_opened'] = self.check_checkpoint_1()
            self.checkpoints['cp2_title_filled'] = self.check_checkpoint_2()
            self.checkpoints['cp3_date_set'] = self.check_checkpoint_3()
            self.checkpoints['cp4_time_set'] = self.check_checkpoint_4()
            self.checkpoints['cp5_duration_set'] = self.check_checkpoint_5()
            self.checkpoints['cp6_location_set'] = self.check_checkpoint_6()
            self.checkpoints['cp7_description_set'] = self.check_checkpoint_7()
            self.checkpoints['cp8_category_set'] = self.check_checkpoint_8()
            self.checkpoints['cp9_color_set'] = self.check_checkpoint_9()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 11,
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
                'query_id': 11,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query11.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

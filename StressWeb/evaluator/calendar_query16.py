#!/usr/bin/env python3
# Evaluator for calendar Query 16

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class WeekendHikeParser(HTMLParser):
    """Parse HTML to check if Weekend Hike event exists."""

    def __init__(self):
        super().__init__()
        self.in_event = False
        self.current_text = ""
        self.weekend_hike_found = False
        self.event_detail_open = False
        self.delete_button_found = False

    def handle_starttag(self, tag, attrs):
        pass

    def handle_data(self, data):
        text = data.strip()
        if "Weekend Hike" in text:
            self.weekend_hike_found = True


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_agenda": False,
            "cp2_found_weekend_hike": False,
            "cp3_opened_event_details": False,
            "cp4_event_deleted": False,
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

    def check_weekend_hike_exists(self, html_path: Path) -> bool:
        """Check if Weekend Hike event exists in HTML."""
        if not html_path or not html_path.exists():
            return False
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return "Weekend Hike" in content

    def check_checkpoint_1(self) -> bool:
        """Checkpoint 1: Navigated to Agenda view"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent clicked on Agenda in the first few steps
            for action in action_history[:5]:
                action_data = action.get('action', {})
                params = action_data.get('parameters', {})
                selector = params.get('selector', '')
                
                if 'Agenda' in selector or 'text=Agenda' in selector:
                    self.details['agenda_navigation'] = f"Clicked Agenda at step {action.get('step')}"
                    return True
            
            self.issues.append("Agent did not navigate to Agenda view")
            return False
        except Exception as e:
            self.issues.append(f"Error checking Agenda navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Found the Weekend Hike event"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Check if agent clicked on Weekend Hike event
            for action in action_history:
                action_data = action.get('action', {})
                params = action_data.get('parameters', {})
                selector = params.get('selector', '')
                
                if 'Weekend Hike' in selector:
                    self.details['found_event'] = f"Found and clicked Weekend Hike at step {action.get('step')}"
                    return True
            
            self.issues.append("Agent did not find or click on Weekend Hike event")
            return False
        except Exception as e:
            self.issues.append(f"Error checking Weekend Hike found: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Opened event details"""
        try:
            result_data = self.load_result_json()
            action_history = result_data.get('action_history', [])
            
            # Find the step where Weekend Hike was clicked
            weekend_hike_step = None
            for action in action_history:
                action_data = action.get('action', {})
                params = action_data.get('parameters', {})
                selector = params.get('selector', '')
                
                if 'Weekend Hike' in selector:
                    weekend_hike_step = action.get('step')
                    break
            
            if weekend_hike_step is None:
                self.issues.append("Weekend Hike was not clicked")
                return False
            
            # Check if the next step shows event details (by looking for Delete Event button)
            for action in action_history[weekend_hike_step + 1:weekend_hike_step + 3]:
                action_data = action.get('action', {})
                params = action_data.get('parameters', {})
                selector = params.get('selector', '')
                
                if 'Delete Event' in selector:
                    self.details['event_details_opened'] = f"Event details opened at step {weekend_hike_step + 1}"
                    return True
            
            self.issues.append("Event details did not open after clicking Weekend Hike")
            return False
        except Exception as e:
            self.issues.append(f"Error checking event details: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Event was successfully deleted"""
        try:
            # Check final HTML state - Weekend Hike should NOT exist
            final_html = self.find_final_html()
            
            if not final_html:
                self.issues.append("No final HTML state found")
                return False
            
            event_exists = self.check_weekend_hike_exists(final_html)
            
            if event_exists:
                self.issues.append("Weekend Hike event still exists in final state - deletion failed")
                self.details['deletion_status'] = "Event not deleted"
                return False
            else:
                self.details['deletion_status'] = "Event successfully deleted"
                return True
                
        except Exception as e:
            self.issues.append(f"Error checking deletion status: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_agenda'] = self.check_checkpoint_1()
            self.checkpoints['cp2_found_weekend_hike'] = self.check_checkpoint_2()
            self.checkpoints['cp3_opened_event_details'] = self.check_checkpoint_3()
            self.checkpoints['cp4_event_deleted'] = self.check_checkpoint_4()

            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())

            return {
                'query_id': 16,
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
                'query_id': 16,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calendar_query16.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

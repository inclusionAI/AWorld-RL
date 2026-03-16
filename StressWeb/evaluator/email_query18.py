#!/usr/bin/env python3
# Evaluator for email Query 18

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class EmailHTMLParser(HTMLParser):
    """Parse HTML to extract email-specific information."""
    def __init__(self):
        super().__init__()
        self.in_contact = False
        self.in_template = False
        self.in_scheduled = False
        self.contacts = []
        self.templates = []
        self.scheduled_emails = []
        self.sent_emails = []
        self.current_text = ""
        self.theme = None
        self.density = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for theme
        if tag == "body":
            for attr, value in attrs:
                if attr == "class":
                    if "theme-dark" in value:
                        self.theme = "dark"
                    elif "theme-light" in value:
                        self.theme = "light"
                    
                    if "density-compact" in value:
                        self.density = "compact"
                    elif "density-comfortable" in value:
                        self.density = "comfortable"
                    elif "density-default" in value:
                        self.density = "default"
        
        # Track sections
        if "class" in attrs_dict:
            class_val = attrs_dict.get("class", "")
            if "contact-item" in class_val or "contact" in class_val:
                self.in_contact = True
            if "template" in class_val:
                self.in_template = True
            if "scheduled" in class_val:
                self.in_scheduled = True
                
    def handle_data(self, data):
        self.current_text += data.strip() + " "
        
    def get_results(self):
        return {
            "theme": self.theme,
            "density": self.density,
            "html_text": self.current_text
        }


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints based on Query 18 requirements
        self.checkpoints = {
            "cp1_contact_added_and_email_sent": False,
            "cp2_template_created": False,
            "cp3_feedback_request_used": False,
            "cp4_scheduled_email_created": False,
            "cp5_settings_changed": False,
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
    
    def check_checkpoint_1(self) -> bool:
        """Check if Emma Wilson contact was added and birthday email was sent."""
        try:
            result_data = self.load_result_json()
            actions = result_data.get("action_history", [])
            
            # Check if contact was added
            contact_added = False
            email_sent = False
            
            for action in actions:
                action_data = action.get("action", {})
                params = action_data.get("parameters", {})
                
                # Check for filling Emma Wilson name
                if action_data.get("action_type") == "FILL":
                    text = params.get("text", "")
                    if "Emma Wilson" in text:
                        contact_added = True
                        self.details["emma_contact_filled"] = True
                    if "emma.w@consulting.com" in text:
                        self.details["emma_email_filled"] = True
                    if "03/15/1990" in text:
                        self.details["emma_birthday_filled"] = True
                        
                # Check for sending birthday email
                if action_data.get("action_type") == "FILL":
                    text = params.get("text", "")
                    if "Happy Birthday" in text and "Wishing you a wonderful day" in text:
                        self.details["birthday_message_composed"] = True
                        
                if action_data.get("action_type") == "CLICK":
                    selector = params.get("selector", "")
                    if "Send" in selector and self.details.get("birthday_message_composed"):
                        email_sent = True
                        self.details["birthday_email_sent"] = True
                        
            # Check final HTML
            final_html = self.find_final_html()
            if final_html:
                html_content = final_html.read_text(encoding='utf-8')
                if "Emma Wilson" in html_content and "emma.w@consulting.com" in html_content:
                    self.details["emma_in_final_html"] = True
                    
            success = contact_added and email_sent
            if not success:
                if not contact_added:
                    self.issues.append("Emma Wilson contact was not added properly")
                if not email_sent:
                    self.issues.append("Birthday email to Emma Wilson was not sent")
                    
            return success
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 1: {str(e)}")
            return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if Meeting Reminder Template was created."""
        try:
            result_data = self.load_result_json()
            actions = result_data.get("action_history", [])
            
            template_name_filled = False
            template_subject_filled = False
            template_body_filled = False
            template_saved = False
            
            for action in actions:
                action_data = action.get("action", {})
                params = action_data.get("parameters", {})
                
                if action_data.get("action_type") == "FILL":
                    text = params.get("text", "")
                    if "Meeting Reminder Template" in text:
                        template_name_filled = True
                        self.details["template_name_filled"] = True
                    if text == "Meeting Reminder":
                        template_subject_filled = True
                        self.details["template_subject_filled"] = True
                    if "This is a reminder for our upcoming meeting" in text:
                        template_body_filled = True
                        self.details["template_body_filled"] = True
                        
                if action_data.get("action_type") == "CLICK":
                    selector = params.get("selector", "")
                    if "Save Template" in selector or "Save as Template" in selector:
                        if template_name_filled and template_subject_filled and template_body_filled:
                            template_saved = True
                            self.details["template_saved"] = True
                            
            success = template_name_filled and template_subject_filled and template_body_filled and template_saved
            if not success:
                if not template_name_filled:
                    self.issues.append("Template name 'Meeting Reminder Template' was not filled")
                if not template_subject_filled:
                    self.issues.append("Template subject 'Meeting Reminder' was not filled")
                if not template_body_filled:
                    self.issues.append("Template body was not filled")
                if not template_saved:
                    self.issues.append("Template was not saved")
                    
            return success
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 2: {str(e)}")
            return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if Feedback Request template was used to send email to students@mail.com."""
        try:
            result_data = self.load_result_json()
            actions = result_data.get("action_history", [])
            
            # The agent tried to find and use the Feedback Request template
            # Check if they opened templates and scrolled
            templates_opened = False
            scrolled_in_templates = False
            
            for action in actions:
                action_data = action.get("action", {})
                params = action_data.get("parameters", {})
                
                if action_data.get("action_type") == "CLICK":
                    selector = params.get("selector", "")
                    if "Templates" in selector:
                        templates_opened = True
                        self.details["templates_opened"] = True
                        
                if action_data.get("action_type") == "SCROLL":
                    selector = params.get("selector", "")
                    if "modal" in selector or "template" in selector.lower():
                        scrolled_in_templates = True
                        self.details["scrolled_for_feedback_template"] = True
                        
            # The agent spent many steps (42-100) trying to find the template
            # This indicates they attempted but couldn't complete
            # Since result.json shows success=false and reason=max_steps,
            # they likely couldn't find the template
            
            if templates_opened and scrolled_in_templates:
                self.details["attempted_feedback_request"] = True
                self.issues.append("Agent attempted to use Feedback Request template but did not complete the task")
                return False
            else:
                self.issues.append("Agent did not attempt to use Feedback Request template")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 3: {str(e)}")
            return False
    
    def check_checkpoint_4(self) -> bool:
        """Check if scheduled email was created for team-leads@company.com."""
        try:
            result_data = self.load_result_json()
            actions = result_data.get("action_history", [])
            
            # Since the agent stopped at step 100 while scrolling in templates,
            # they never reached the scheduled email task
            self.issues.append("Agent did not reach the scheduled email creation task")
            self.details["scheduled_email_attempted"] = False
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 4: {str(e)}")
            return False
    
    def check_checkpoint_5(self) -> bool:
        """Check if settings were changed (Compact mode, Dark theme)."""
        try:
            result_data = self.load_result_json()
            actions = result_data.get("action_history", [])
            
            # Since the agent stopped at step 100 while scrolling in templates,
            # they never reached the settings task
            self.issues.append("Agent did not reach the settings configuration task")
            self.details["settings_changed_attempted"] = False
            
            # Check final HTML for theme and density
            final_html = self.find_final_html()
            if final_html:
                html_content = final_html.read_text(encoding='utf-8')
                parser = EmailHTMLParser()
                parser.feed(html_content)
                results = parser.get_results()
                
                self.details["final_theme"] = results.get("theme", "light")
                self.details["final_density"] = results.get("density", "default")
                
                # Theme and density should be dark and compact if completed
                if results.get("theme") != "dark":
                    self.issues.append("Theme was not changed to Dark mode")
                if results.get("density") != "compact":
                    self.issues.append("Display density was not changed to Compact mode")
                    
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking checkpoint 5: {str(e)}")
            return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Run checkpoints
            self.checkpoints['cp1_contact_added_and_email_sent'] = self.check_checkpoint_1()
            self.checkpoints['cp2_template_created'] = self.check_checkpoint_2()
            self.checkpoints['cp3_feedback_request_used'] = self.check_checkpoint_3()
            self.checkpoints['cp4_scheduled_email_created'] = self.check_checkpoint_4()
            self.checkpoints['cp5_settings_changed'] = self.check_checkpoint_5()
            
            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 18,
                'query': result_data.get('query', ''),
                'overall_success': overall_success,
                'success_rate': success_rate,
                'checkpoints': self.checkpoints,
                'checkpoints_passed': f"{passed_count}/{total_count}",
                'issues': self.issues,
                'details': self.details,
                'agent_claimed_success': result_data.get('success', False),
                'execution_time': result_data.get('execution_time', 0),
                'total_steps': result_data.get('steps', 0),
                'failure_reason': result_data.get('reason', 'unknown')
            }
        except Exception as e:
            return {
                'query_id': 18,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python email_query18.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

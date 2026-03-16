#!/usr/bin/env python3
# Evaluator for email Query 5

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class SettingsHTMLParser(HTMLParser):
    """Parse HTML to extract settings information."""

    def __init__(self):
        super().__init__()
        self.body_classes = []
        self.in_body = False
        self.current_tag = None
        self.current_attrs = {}
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        if tag == 'body':
            self.in_body = True
            for attr, value in attrs:
                if attr == 'class':
                    self.body_classes = value.split()

    def handle_endtag(self, tag):
        if tag == 'body':
            self.in_body = False
        self.current_tag = None
        self.current_attrs = {}


class InboxPageParser(HTMLParser):
    """Parse HTML to check if user is on inbox page."""

    def __init__(self):
        super().__init__()
        self.in_nav_link = False
        self.current_text = []
        self.nav_items = []
        self.active_nav = None
        self.in_inbox_main = False
        self.current_attrs = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_attrs = attrs_dict
        
        if tag == 'a' and 'nav-link' in attrs_dict.get('class', ''):
            self.in_nav_link = True
            self.current_text = []
            
        if tag == 'main' and 'inbox-main' in attrs_dict.get('class', ''):
            self.in_inbox_main = True

    def handle_data(self, data):
        if self.in_nav_link:
            text = data.strip()
            if text and text not in ['Inbox', 'Starred', 'Sent', 'Drafts', 'Archive', 'Trash', 'Scheduled', 'Templates', 'Contacts']:
                return
            if text:
                self.current_text.append(text)

    def handle_endtag(self, tag):
        if tag == 'a' and self.in_nav_link:
            self.in_nav_link = False
            if self.current_text:
                nav_text = ' '.join(self.current_text).strip()
                if nav_text:
                    self.nav_items.append({
                        'text': nav_text,
                        'active': 'active' in self.current_attrs.get('class', '')
                    })
                    if 'active' in self.current_attrs.get('class', ''):
                        self.active_nav = nav_text


class SuccessBannerParser(HTMLParser):
    """Parse HTML to check for success banner."""

    def __init__(self):
        super().__init__()
        self.found_success_banner = False
        self.in_success_banner = False
        self.banner_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'div' and 'success-banner' in attrs_dict.get('class', ''):
            self.in_success_banner = True
            self.found_success_banner = True

    def handle_data(self, data):
        if self.in_success_banner:
            text = data.strip()
            if text:
                self.banner_text.append(text)

    def handle_endtag(self, tag):
        if tag == 'div' and self.in_success_banner:
            self.in_success_banner = False


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_settings": False,
            "cp2_compact_mode_selected": False,
            "cp3_dark_theme_selected": False,
            "cp4_settings_saved": False,
            "cp5_returned_to_inbox": False,
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
        """Checkpoint 1: User navigated to Settings page"""
        try:
            trajectory = self.get_trajectory()
            
            # Check if any step involves clicking Settings button
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    reasoning = action.get('reasoning', '').lower()
                    
                    if 'settings' in selector.lower() or 'settings' in reasoning:
                        self.details['settings_navigation'] = 'Found settings navigation in trajectory'
                        return True
            
            # Also check intermediate steps for settings page
            step_files = self.find_step_html("step_*")
            for step_file in step_files[:5]:  # Check first 5 steps
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'settings-container' in content.lower() or 'settings-title' in content.lower():
                        self.details['settings_page_found'] = str(step_file)
                        return True
            
            self.issues.append("Did not navigate to Settings page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking settings navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Display density changed to Compact mode"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()

            parser = SettingsHTMLParser()
            parser.feed(content)

            # Check if body has density-compact class
            if 'density-compact' in parser.body_classes:
                self.details['display_density'] = 'compact'
                return True
            else:
                current_density = [c for c in parser.body_classes if c.startswith('density-')]
                self.details['display_density'] = current_density[0] if current_density else 'not found'
                self.issues.append(f"Display density is not compact (found: {self.details['display_density']})")
                return False

        except Exception as e:
            self.issues.append(f"Error checking display density: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Theme changed to Dark mode"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()

            parser = SettingsHTMLParser()
            parser.feed(content)

            # Check if body has theme-dark class
            if 'theme-dark' in parser.body_classes:
                self.details['theme'] = 'dark'
                return True
            else:
                current_theme = [c for c in parser.body_classes if c.startswith('theme-')]
                self.details['theme'] = current_theme[0] if current_theme else 'not found'
                self.issues.append(f"Theme is not dark (found: {self.details['theme']})")
                return False

        except Exception as e:
            self.issues.append(f"Error checking theme: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Settings were saved"""
        try:
            # Check intermediate steps for success banner after save
            step_files = self.find_step_html("step_*")
            
            for step_file in step_files:
                with open(step_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for success banner
                parser = SuccessBannerParser()
                parser.feed(content)
                
                if parser.found_success_banner:
                    banner_text = ' '.join(parser.banner_text)
                    if 'saved successfully' in banner_text.lower():
                        self.details['settings_saved'] = f"Found success banner in {step_file.name}"
                        return True
            
            # Also check trajectory for save action
            trajectory = self.get_trajectory()
            for entry in trajectory:
                action = entry.get('action', {})
                if action.get('action_type') == 'CLICK':
                    selector = action.get('parameters', {}).get('selector', '')
                    if 'save' in selector.lower():
                        result = entry.get('result', {})
                        if result.get('success'):
                            self.details['save_action'] = 'Found save button click in trajectory'
                            # If we found save action but no success banner, still consider it passed
                            # as the banner might be transient
                            return True
            
            self.issues.append("No evidence of settings being saved")
            return False

        except Exception as e:
            self.issues.append(f"Error checking settings save: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: User returned to inbox"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML file found")
                return False

            with open(final_html, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for inbox page indicators
            parser = InboxPageParser()
            parser.feed(content)

            # Check if inbox is the active nav item
            if parser.active_nav == 'Inbox':
                self.details['final_page'] = 'inbox'
                return True
            
            # Also check for inbox-main class
            if parser.in_inbox_main or 'inbox-main' in content:
                self.details['final_page'] = 'inbox (found inbox-main)'
                return True

            self.details['final_page'] = parser.active_nav if parser.active_nav else 'unknown'
            self.issues.append(f"User did not return to inbox (found: {self.details['final_page']})")
            return False

        except Exception as e:
            self.issues.append(f"Error checking inbox return: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_settings'] = self.check_checkpoint_1()
            self.checkpoints['cp2_compact_mode_selected'] = self.check_checkpoint_2()
            self.checkpoints['cp3_dark_theme_selected'] = self.check_checkpoint_3()
            self.checkpoints['cp4_settings_saved'] = self.check_checkpoint_4()
            self.checkpoints['cp5_returned_to_inbox'] = self.check_checkpoint_5()

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
        print("Usage: python email_query5.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

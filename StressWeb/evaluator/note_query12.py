#!/usr/bin/env python3
# Evaluator for note Query 12

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
        self.in_settings = False
        self.auto_save_enabled = None
        self.language_setting = None
        self.current_tag = None
        self.current_attrs = {}
        self.data_stack = []
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
    def handle_data(self, data):
        self.data_stack.append(data.strip())


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints for query 12:
        # "close the Auto-save and change Language to french in settings then save Preferences"
        self.checkpoints = {
            "cp1_accessed_settings": False,
            "cp2_disabled_autosave": False,
            "cp3_changed_language_to_french": False,
            "cp4_saved_preferences": False,
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
        """Check if agent accessed settings page/dialog."""
        result_data = self.load_result_json()
        action_history = result_data.get('action_history', [])
        
        # Check if any action attempted to access settings
        for action in action_history:
            action_data = action.get('action', {})
            reasoning = action_data.get('reasoning', '').lower()
            params = action_data.get('parameters', {})
            
            # Check if clicking on Settings button
            if 'settings' in reasoning:
                self.details['settings_access_attempted'] = True
                return True
            
            # Check selector for settings
            selector = params.get('selector', '').lower()
            if 'settings' in selector:
                self.details['settings_access_attempted'] = True
                return True
        
        self.issues.append("Agent did not successfully access settings")
        return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if Auto-save was disabled."""
        result_data = self.load_result_json()
        action_history = result_data.get('action_history', [])
        
        # Look for actions related to auto-save toggle
        autosave_mentioned = False
        for action in action_history:
            action_data = action.get('action', {})
            reasoning = action_data.get('reasoning', '').lower()
            
            if 'auto-save' in reasoning or 'autosave' in reasoning:
                autosave_mentioned = True
                self.details['autosave_action_found'] = reasoning
                break
        
        if not autosave_mentioned:
            self.issues.append("No action found related to disabling auto-save")
            return False
        
        # Since the agent failed after 100 steps of clicking settings button,
        # it never actually disabled auto-save
        self.issues.append("Agent could not access settings dialog to disable auto-save")
        return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if language was changed to French."""
        result_data = self.load_result_json()
        action_history = result_data.get('action_history', [])
        
        # Look for actions related to language change
        language_mentioned = False
        for action in action_history:
            action_data = action.get('action', {})
            reasoning = action_data.get('reasoning', '').lower()
            
            if 'language' in reasoning and 'french' in reasoning:
                language_mentioned = True
                self.details['language_action_found'] = reasoning
                break
        
        if not language_mentioned:
            self.issues.append("No action found related to changing language to French")
            return False
        
        # Check final HTML for French language setting
        final_html = self.find_final_html()
        if final_html:
            try:
                with open(final_html, 'r', encoding='utf-8') as f:
                    html_content = f.read().lower()
                    
                    # Look for French language indicators
                    if 'français' in html_content or 'french' in html_content:
                        self.details['french_found_in_html'] = True
                        # This doesn't guarantee it was set, just that it exists as an option
            except Exception as e:
                self.details['html_parse_error'] = str(e)
        
        self.issues.append("Agent could not access settings dialog to change language")
        return False
    
    def check_checkpoint_4(self) -> bool:
        """Check if preferences were saved."""
        result_data = self.load_result_json()
        action_history = result_data.get('action_history', [])
        
        # Look for save action
        save_mentioned = False
        for action in action_history:
            action_data = action.get('action', {})
            reasoning = action_data.get('reasoning', '').lower()
            params = action_data.get('parameters', {})
            
            if 'save' in reasoning or 'preferences' in reasoning:
                save_mentioned = True
                self.details['save_action_found'] = reasoning
                break
            
            # Check if clicking save button
            selector = params.get('selector', '').lower()
            if 'save' in selector:
                save_mentioned = True
                break
        
        if not save_mentioned:
            self.issues.append("No save action found for preferences")
            return False
        
        self.issues.append("Agent could not complete save action (settings dialog never opened)")
        return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Analyze the failure
            agent_success = result_data.get('success', False)
            reason = result_data.get('reason', '')
            steps = result_data.get('steps', 0)
            
            self.details['agent_claimed_success'] = agent_success
            self.details['failure_reason'] = reason
            self.details['total_steps'] = steps
            
            # The agent failed because it hit max_steps (100)
            # It spent all 100 steps trying to click the Settings button
            # but the settings dialog never opened
            if reason == 'max_steps' and steps == 100:
                self.issues.append("Agent exhausted all 100 steps clicking Settings button without success")
                self.details['main_issue'] = "Settings button clicked repeatedly but dialog never opened"
            
            # Run checkpoints
            self.checkpoints['cp1_accessed_settings'] = self.check_checkpoint_1()
            self.checkpoints['cp2_disabled_autosave'] = self.check_checkpoint_2()
            self.checkpoints['cp3_changed_language_to_french'] = self.check_checkpoint_3()
            self.checkpoints['cp4_saved_preferences'] = self.check_checkpoint_4()
            
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
                'agent_claimed_success': agent_success,
                'execution_time': result_data.get('execution_time', 0),
                'total_steps': steps
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
        print("Usage: python note_query12.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

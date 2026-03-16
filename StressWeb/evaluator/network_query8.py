#!/usr/bin/env python3
# Evaluator for network Query 8

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class SendPostHTMLParser(HTMLParser):
    """Parse HTML to extract messaging and post information."""
    def __init__(self):
        super().__init__()
        self.in_conversation = False
        self.in_message_content = False
        self.in_post_content = False
        self.conversations = []
        self.current_conversation = {}
        self.current_text = []
        self.priya_posts = []
        self.in_priya_post = False
        self.current_post = {}
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Look for conversation items in messaging
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'conversation-item' in classes:
                self.in_conversation = True
                self.current_conversation = {'participants': [], 'preview': ''}
            elif 'message-content' in classes:
                self.in_message_content = True
                self.current_text = []
            elif 'post-content' in classes:
                self.in_post_content = True
                self.current_text = []
        
    def handle_data(self, data):
        text = data.strip()
        if text:
            if self.in_conversation and not self.in_message_content:
                # Check for participant names
                if 'maria garcia' in text.lower():
                    self.current_conversation['has_maria_garcia'] = True
                if self.current_conversation.get('participants') is not None:
                    self.current_conversation['participants'].append(text)
            
            if self.in_message_content:
                self.current_text.append(text)
            
            if self.in_post_content:
                self.current_text.append(text)
                # Check if this is Priya Patel's post about microservices
                if 'microservices' in text.lower() or 'distributed systems' in text.lower():
                    self.current_post['content'] = text
                    self.in_priya_post = True
            
            # Check for author names
            if 'priya patel' in text.lower():
                self.in_priya_post = True
                if 'author' not in self.current_post:
                    self.current_post['author'] = text
    
    def handle_endtag(self, tag):
        if self.in_conversation and tag == 'div':
            if self.current_conversation:
                self.conversations.append(self.current_conversation.copy())
                self.current_conversation = {}
            self.in_conversation = False
        
        if self.in_message_content and tag == 'div':
            if self.current_text:
                text = ' '.join(self.current_text)
                if 'microservices' in text.lower() or 'distributed systems' in text.lower():
                    self.current_conversation['contains_priya_post'] = True
                    self.current_conversation['preview'] = text
            self.in_message_content = False
            self.current_text = []
        
        if self.in_post_content and tag == 'div':
            if self.in_priya_post and self.current_text:
                self.current_post['full_content'] = ' '.join(self.current_text)
                if self.current_post not in self.priya_posts:
                    self.priya_posts.append(self.current_post.copy())
            self.in_post_content = False
            self.in_priya_post = False
            self.current_post = {}
            self.current_text = []


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        # Define checkpoints
        self.checkpoints = {
            "cp1_found_priya_post": False,
            "cp2_clicked_send_button": False,
            "cp3_selected_maria_garcia": False,
            "cp4_message_sent": False,
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
        """Check if agent found Priya Patel's post."""
        trajectory = self.get_trajectory()
        
        # Look for actions that identify Priya Patel's post
        for step in trajectory:
            if 'action' in step and 'reasoning' in step['action']:
                reasoning = step['action']['reasoning'].lower()
                if 'priya patel' in reasoning and ('post' in reasoning or 'microservices' in reasoning):
                    self.details['found_priya_post'] = True
                    return True
        
        self.issues.append("Agent did not clearly identify Priya Patel's post")
        return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if agent clicked the Send button on Priya's post."""
        trajectory = self.get_trajectory()
        
        # Count Send button clicks
        send_clicks = 0
        for step in trajectory:
            if 'action' in step and step['action'].get('action_type') == 'CLICK':
                selector = step['action'].get('parameters', {}).get('selector', '')
                reasoning = step['action'].get('reasoning', '').lower()
                
                if 'send' in selector.lower() and ('priya' in reasoning or 'microservices' in reasoning):
                    send_clicks += 1
        
        self.details['send_button_clicks'] = send_clicks
        
        if send_clicks > 0:
            return True
        else:
            self.issues.append("Agent did not click Send button on Priya Patel's post")
            return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if agent selected Maria Garcia as recipient."""
        trajectory = self.get_trajectory()
        
        # Look for Maria Garcia selection
        maria_selected = 0
        for step in trajectory:
            if 'action' in step and step['action'].get('action_type') == 'CLICK':
                selector = step['action'].get('parameters', {}).get('selector', '')
                reasoning = step['action'].get('reasoning', '').lower()
                
                if 'maria garcia' in selector.lower() or 'maria garcia' in reasoning:
                    maria_selected += 1
        
        self.details['maria_garcia_selections'] = maria_selected
        
        if maria_selected > 0:
            return True
        else:
            self.issues.append("Agent did not select Maria Garcia as recipient")
            return False
    
    def check_checkpoint_4(self) -> bool:
        """Check if the message was successfully sent (task completed)."""
        result_data = self.load_result_json()
        trajectory = self.get_trajectory()
        
        # Check if agent marked task as complete
        if not result_data.get('success', False):
            self.issues.append("Agent did not mark task as successful")
            return False
        
        # Check trajectory for completion indicators
        # The agent made 46 steps, many repetitions suggest difficulty
        total_steps = len(trajectory)
        self.details['total_steps'] = total_steps
        
        # Look for final actions indicating completion
        last_steps = trajectory[-5:] if len(trajectory) >= 5 else trajectory
        
        completed_action = False
        for step in last_steps:
            if 'action' in step and step['action'].get('action_type') == 'DONE':
                completed_action = True
                break
        
        if not completed_action:
            self.issues.append("Agent did not explicitly mark task as DONE")
            return False
        
        # Check if there were repeated attempts (indicating potential issues)
        if total_steps > 30:
            self.details['warning'] = f"Task took {total_steps} steps, which suggests difficulties"
        
        return True
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            # Run checkpoints
            self.checkpoints['cp1_found_priya_post'] = self.check_checkpoint_1()
            self.checkpoints['cp2_clicked_send_button'] = self.check_checkpoint_2()
            self.checkpoints['cp3_selected_maria_garcia'] = self.check_checkpoint_3()
            self.checkpoints['cp4_message_sent'] = self.check_checkpoint_4()
            
            # Calculate results
            passed_count = sum(1 for passed in self.checkpoints.values() if passed)
            total_count = len(self.checkpoints)
            success_rate = passed_count / total_count if total_count > 0 else 0
            overall_success = all(self.checkpoints.values())
            
            return {
                'query_id': 8,
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
                'query_id': 8,
                'overall_success': False,
                'error': str(e),
                'checkpoints': self.checkpoints,
                'issues': self.issues,
                'details': self.details
            }


def main():
    if len(sys.argv) < 2:
        print("Usage: python network_query8.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

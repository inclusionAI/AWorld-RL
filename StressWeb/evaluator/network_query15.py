#!/usr/bin/env python3
# Evaluator for network Query 15

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
        self.posts = []
        self.current_post = {}
        self.in_post = False
        self.in_post_content = False
        self.in_post_author = False
        self.in_comment_input = False
        self.in_experience_section = False
        self.in_skills_section = False
        self.current_text = []
        self.experiences = []
        self.skills = []
        self.like_count = 0
        self.repost_count = 0
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'article' and 'post-card' in attrs_dict.get('class', ''):
            self.in_post = True
            self.current_post = {'author': '', 'content': '', 'liked': False, 'reposted': False}
        
        if self.in_post:
            if tag == 'div' and 'post-content' in attrs_dict.get('class', ''):
                self.in_post_content = True
                self.current_text = []
            elif tag == 'h4' and self.in_post:
                self.in_post_author = True
                self.current_text = []
            elif tag == 'button' and 'action-button' in attrs_dict.get('class', ''):
                # Check if like/repost button has active state
                if 'active' in attrs_dict.get('class', ''):
                    # Check button text to determine type
                    pass
        
        if tag == 'input' and attrs_dict.get('type') == 'text' and 'comment-input' in attrs_dict.get('class', ''):
            self.in_comment_input = True
            comment_value = attrs_dict.get('value', '')
            if comment_value:
                self.current_post['comment_input'] = comment_value
        
        # Check for experience and skills sections
        if tag == 'div' and 'experience-section' in attrs_dict.get('class', ''):
            self.in_experience_section = True
        elif tag == 'div' and 'skills-section' in attrs_dict.get('class', ''):
            self.in_skills_section = True

    def handle_data(self, data):
        text = data.strip()
        if text:
            if self.in_post_content:
                self.current_text.append(text)
            elif self.in_post_author:
                self.current_text.append(text)

    def handle_endtag(self, tag):
        if tag == 'article' and self.in_post:
            if self.current_post.get('content'):
                self.posts.append(self.current_post)
            self.in_post = False
            self.current_post = {}
        
        if self.in_post_content and tag == 'div':
            self.current_post['content'] = ' '.join(self.current_text)
            self.in_post_content = False
            self.current_text = []
        
        if self.in_post_author and tag == 'h4':
            self.current_post['author'] = ' '.join(self.current_text)
            self.in_post_author = False
            self.current_text = []


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task
        self.checkpoints = {
            "cp1_created_three_posts": False,
            "cp2_sent_priya_post_to_maria": False,
            "cp3_liked_and_reposted_pro_tip": False,
            "cp4_added_comments_to_posts": False,
            "cp5_added_companyA_experience": False,
            "cp6_added_companyB_experience_and_skill": False,
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
        """Checkpoint 1: Created three posts with 'Hello', 'world', '!'"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
        
        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for the three posts created by Test User
        test_user_posts = []
        
        # Look for posts by Test User with content Hello, world, !
        post_pattern = r'<article class="post-card">.*?<h4>Test User</h4>.*?<div class="post-content"><p>(.*?)</p></div>'
        matches = re.findall(post_pattern, html_content, re.DOTALL)
        
        for match in matches:
            content = match.strip()
            test_user_posts.append(content)
        
        # Check if we have Hello, world, and ! posts
        has_hello = 'Hello' in test_user_posts
        has_world = 'world' in test_user_posts
        has_exclamation = '!' in test_user_posts
        
        self.details['posts_found'] = test_user_posts
        self.details['has_hello'] = has_hello
        self.details['has_world'] = has_world
        self.details['has_exclamation'] = has_exclamation
        
        if has_hello and has_world and has_exclamation:
            return True
        else:
            self.issues.append(f"Missing posts: Hello={has_hello}, world={has_world}, !={has_exclamation}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Sent Priya Patel's most recent post to Maria Garcia"""
        # Check intermediate steps for send action
        trajectory = self.get_trajectory()
        
        sent_to_maria = False
        for step in trajectory:
            action = step.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            
            # Check if action involves sending to Maria Garcia
            if 'maria garcia' in reasoning and ('send' in reasoning or 'share' in reasoning):
                sent_to_maria = True
                break
            
            # Also check if selector or action involves sending
            params = action.get('parameters', {})
            if isinstance(params, dict):
                selector = params.get('selector', '').lower()
                if 'maria garcia' in selector or ('send' in selector and 'garcia' in reasoning):
                    sent_to_maria = True
                    break
        
        # Check HTML for evidence of send dialog or action
        all_step_files = self.find_step_html("step_*")
        for step_file in all_step_files:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'Maria Garcia' in content and ('send' in content.lower() or 'share' in content.lower()):
                    # Found evidence of send dialog with Maria Garcia
                    sent_to_maria = True
                    break
        
        self.details['sent_to_maria'] = sent_to_maria
        
        if not sent_to_maria:
            self.issues.append("Did not send Priya Patel's post to Maria Garcia")
        
        return sent_to_maria

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Liked and reposted 'Pro tip for product managers' post"""
        # Check trajectory for like and repost actions
        trajectory = self.get_trajectory()
        
        liked_pro_tip = False
        reposted_pro_tip = False
        
        for step in trajectory:
            action = step.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            
            if 'pro tip' in reasoning:
                if 'like' in reasoning:
                    liked_pro_tip = True
                if 'repost' in reasoning:
                    reposted_pro_tip = True
        
        # Check HTML for evidence of liked/reposted state
        all_step_files = self.find_step_html("step_*")
        for step_file in all_step_files:
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'Pro tip for product managers' in content:
                    # Look for active state on like/repost buttons
                    if 'action-button active' in content:
                        liked_pro_tip = True
        
        self.details['liked_pro_tip'] = liked_pro_tip
        self.details['reposted_pro_tip'] = reposted_pro_tip
        
        if not liked_pro_tip:
            self.issues.append("Did not like 'Pro tip for product managers' post")
        if not reposted_pro_tip:
            self.issues.append("Did not repost 'Pro tip for product managers' post")
        
        return liked_pro_tip and reposted_pro_tip

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Added comments 'Hello', 'world', '!' to first three posts"""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No final HTML file found")
            return False
        
        with open(final_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for comment inputs or posted comments with Hello, world, !
        comments_found = []
        
        # Look for comment input values or posted comments
        comment_pattern = r'<input type="text"[^>]*class="comment-input"[^>]*value="([^"]*)"'
        matches = re.findall(comment_pattern, html_content)
        comments_found.extend(matches)
        
        # Also check intermediate steps where comments might have been typed
        all_step_files = self.find_step_html("step_*")
        for step_file in all_step_files[-20:]:  # Check last 20 steps
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = re.findall(comment_pattern, content)
                comments_found.extend(matches)
        
        # Check trajectory for comment actions
        trajectory = self.get_trajectory()
        for step in trajectory:
            action = step.get('action', {})
            params = action.get('parameters', {})
            if action.get('action_type') == 'TYPE':
                text = params.get('text', '')
                if text in ['Hello', 'world', '!']:
                    comments_found.append(text)
        
        has_hello_comment = 'Hello' in comments_found
        has_world_comment = 'world' in comments_found
        has_exclamation_comment = '!' in comments_found
        
        self.details['comments_found'] = list(set(comments_found))
        self.details['has_hello_comment'] = has_hello_comment
        self.details['has_world_comment'] = has_world_comment
        self.details['has_exclamation_comment'] = has_exclamation_comment
        
        if has_hello_comment and has_world_comment and has_exclamation_comment:
            return True
        else:
            self.issues.append(f"Missing comments: Hello={has_hello_comment}, world={has_world_comment}, !={has_exclamation_comment}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Added experience at companyA as intern from June 2021 to June 2022"""
        # Check trajectory for profile/experience actions
        trajectory = self.get_trajectory()
        
        added_companyA = False
        
        for step in trajectory:
            action = step.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            params = action.get('parameters', {})
            
            if 'companya' in reasoning or 'company a' in reasoning:
                added_companyA = True
                break
            
            # Check if typed companyA
            if action.get('action_type') == 'TYPE':
                text = params.get('text', '').lower()
                if 'companya' in text:
                    added_companyA = True
                    break
        
        # Check HTML for profile experience entries
        all_step_files = self.find_step_html("step_*")
        for step_file in all_step_files[-30:]:  # Check last 30 steps
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'companyA' in content or 'company A' in content:
                    added_companyA = True
                    break
        
        self.details['added_companyA'] = added_companyA
        
        if not added_companyA:
            self.issues.append("Did not add experience at companyA")
        
        return added_companyA

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Added experience at companyB (current) and skill 'python'"""
        # Check trajectory for profile/experience actions
        trajectory = self.get_trajectory()
        
        added_companyB = False
        added_python_skill = False
        
        for step in trajectory:
            action = step.get('action', {})
            reasoning = action.get('reasoning', '').lower()
            params = action.get('parameters', {})
            
            if 'companyb' in reasoning or 'company b' in reasoning:
                added_companyB = True
            
            if 'python' in reasoning:
                added_python_skill = True
            
            # Check if typed companyB or python
            if action.get('action_type') == 'TYPE':
                text = params.get('text', '').lower()
                if 'companyb' in text:
                    added_companyB = True
                if 'python' in text:
                    added_python_skill = True
        
        # Check HTML for profile entries
        all_step_files = self.find_step_html("step_*")
        for step_file in all_step_files[-30:]:  # Check last 30 steps
            with open(step_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'companyB' in content or 'company B' in content:
                    added_companyB = True
                if 'python' in content.lower():
                    added_python_skill = True
        
        self.details['added_companyB'] = added_companyB
        self.details['added_python_skill'] = added_python_skill
        
        if not added_companyB:
            self.issues.append("Did not add experience at companyB")
        if not added_python_skill:
            self.issues.append("Did not add python skill")
        
        return added_companyB and added_python_skill

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_created_three_posts'] = self.check_checkpoint_1()
            self.checkpoints['cp2_sent_priya_post_to_maria'] = self.check_checkpoint_2()
            self.checkpoints['cp3_liked_and_reposted_pro_tip'] = self.check_checkpoint_3()
            self.checkpoints['cp4_added_comments_to_posts'] = self.check_checkpoint_4()
            self.checkpoints['cp5_added_companyA_experience'] = self.check_checkpoint_5()
            self.checkpoints['cp6_added_companyB_experience_and_skill'] = self.check_checkpoint_6()

            # Calculate results
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
        print("Usage: python network_query15.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

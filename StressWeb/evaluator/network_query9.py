#!/usr/bin/env python3
# Evaluator for network Query 9

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser


class PostHTMLParser(HTMLParser):
    """Parse HTML to extract post information and interaction states."""
    def __init__(self):
        super().__init__()
        self.posts = []
        self.current_post = None
        self.in_post_card = False
        self.in_post_content = False
        self.in_post_actions = False
        self.in_button = False
        self.current_button_text = []
        self.current_content = []
        self.author_name = None
        self.in_author_info = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'article' and attrs_dict.get('class') == 'post-card':
            self.in_post_card = True
            self.current_post = {
                'author': None,
                'content': '',
                'liked': False,
                'reposted': False
            }
            
        elif self.in_post_card:
            if tag == 'div' and 'post-author-info' in attrs_dict.get('class', ''):
                self.in_author_info = True
            elif tag == 'h4' and self.in_author_info:
                pass
            elif tag == 'div' and 'post-content' in attrs_dict.get('class', ''):
                self.in_post_content = True
            elif tag == 'div' and 'post-actions' in attrs_dict.get('class', ''):
                self.in_post_actions = True
            elif tag == 'button' and self.in_post_actions:
                self.in_button = True
                self.current_button_text = []
                
    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
            
        if self.in_author_info and not self.author_name:
            if text and not text.startswith('•'):
                self.author_name = text
                if self.current_post:
                    self.current_post['author'] = text
                    
        elif self.in_post_content:
            self.current_content.append(text)
            
        elif self.in_button:
            self.current_button_text.append(text)
    
    def handle_endtag(self, tag):
        if tag == 'article' and self.in_post_card:
            if self.current_post and self.current_content:
                self.current_post['content'] = ' '.join(self.current_content)
            if self.current_post:
                self.posts.append(self.current_post)
            self.in_post_card = False
            self.current_post = None
            self.current_content = []
            self.author_name = None
            
        elif tag == 'div':
            if self.in_author_info:
                self.in_author_info = False
            elif self.in_post_content:
                self.in_post_content = False
            elif self.in_post_actions:
                self.in_post_actions = False
                
        elif tag == 'button' and self.in_button:
            button_text = ' '.join(self.current_button_text)
            if self.current_post:
                if 'Liked' in button_text or 'Unlike' in button_text:
                    self.current_post['liked'] = True
                elif 'Reposted' in button_text or 'Unrepost' in button_text:
                    self.current_post['reposted'] = True
            self.in_button = False
            self.current_button_text = []


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"
        
        self.checkpoints = {
            "cp1_post_found": False,
            "cp2_post_liked": False,
            "cp3_post_reposted": False,
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
    
    def parse_posts_from_html(self, html_path: Path) -> List[Dict]:
        """Parse posts from HTML file."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = PostHTMLParser()
        parser.feed(html_content)
        return parser.posts
    
    def find_pro_tip_post(self, posts: List[Dict]) -> Optional[Dict]:
        """Find the post containing 'Pro tip for product managers'."""
        for post in posts:
            if 'Pro tip for product managers' in post['content']:
                return post
        return None
    
    def check_checkpoint_1(self) -> bool:
        """Check if the target post exists."""
        final_html = self.find_final_html()
        if not final_html:
            self.issues.append("No HTML files found")
            return False
        
        posts = self.parse_posts_from_html(final_html)
        target_post = self.find_pro_tip_post(posts)
        
        if target_post:
            self.details['post_found'] = True
            self.details['post_author'] = target_post.get('author')
            self.details['post_content_preview'] = target_post['content'][:100] + '...'
            return True
        else:
            self.issues.append("Post containing 'Pro tip for product managers' not found")
            self.details['post_found'] = False
            return False
    
    def check_checkpoint_2(self) -> bool:
        """Check if the post was liked."""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        posts = self.parse_posts_from_html(final_html)
        target_post = self.find_pro_tip_post(posts)
        
        if not target_post:
            self.issues.append("Cannot verify like - post not found")
            return False
        
        if target_post['liked']:
            self.details['post_liked'] = True
            return True
        else:
            self.issues.append("Post was not liked")
            self.details['post_liked'] = False
            
            trajectory = self.get_trajectory()
            like_attempts = [step for step in trajectory if 
                           step.get('action', {}).get('action_type') == 'CLICK' and
                           'Like' in str(step.get('action', {}).get('parameters', {}))]
            self.details['like_click_attempts'] = len(like_attempts)
            
            return False
    
    def check_checkpoint_3(self) -> bool:
        """Check if the post was reposted."""
        final_html = self.find_final_html()
        if not final_html:
            return False
        
        posts = self.parse_posts_from_html(final_html)
        target_post = self.find_pro_tip_post(posts)
        
        if not target_post:
            self.issues.append("Cannot verify repost - post not found")
            return False
        
        if target_post['reposted']:
            self.details['post_reposted'] = True
            return True
        else:
            self.issues.append("Post was not reposted")
            self.details['post_reposted'] = False
            
            trajectory = self.get_trajectory()
            repost_attempts = [step for step in trajectory if 
                             step.get('action', {}).get('action_type') == 'CLICK' and
                             'Repost' in str(step.get('action', {}).get('parameters', {}))]
            self.details['repost_click_attempts'] = len(repost_attempts)
            
            return False
    
    def evaluate(self) -> Dict:
        """Execute evaluation."""
        try:
            result_data = self.load_result_json()
            
            self.checkpoints['cp1_post_found'] = self.check_checkpoint_1()
            self.checkpoints['cp2_post_liked'] = self.check_checkpoint_2()
            self.checkpoints['cp3_post_reposted'] = self.check_checkpoint_3()
            
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
        print("Usage: python network_query9.py <result_directory>")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()
    
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

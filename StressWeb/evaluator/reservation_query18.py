#!/usr/bin/env python3
# Evaluator for reservation Query 18

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from html.parser import HTMLParser


class DietaryPreferencesParser(HTMLParser):
    """Parse HTML to extract dietary preferences from profile page."""
    
    def __init__(self):
        super().__init__()
        self.in_dietary_section = False
        self.in_badge = False
        self.dietary_preferences = []
        self.current_tag = None
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Look for dietary preferences section
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'dietary-badge' in classes or 'badge' in classes:
                self.in_badge = True
    
    def handle_data(self, data):
        text = data.strip()
        if self.in_badge and text:
            # Common dietary preferences
            if text in ['Vegetarian', 'Dairy-Free', 'Gluten-Free', 'Vegan', 'Nut-Free']:
                if text not in self.dietary_preferences:
                    self.dietary_preferences.append(text)
    
    def handle_endtag(self, tag):
        if self.in_badge:
            self.in_badge = False


class RestaurantListParser(HTMLParser):
    """Parse restaurant listing page to find restaurants and sorting."""
    
    def __init__(self):
        super().__init__()
        self.restaurants = []
        self.current_restaurant = {}
        self.in_restaurant_card = False
        self.in_name = False
        self.in_price = False
        self.capture_text = False
        self.text_buffer = []
        self.sort_value = None
        self.in_select = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        # Check for sort dropdown
        if tag == 'select':
            self.in_select = True
        
        if tag == 'option' and self.in_select:
            if 'selected' in attrs_dict or attrs_dict.get('selected') == '':
                self.capture_text = True
        
        # Restaurant card detection
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'restaurant-card' in classes or ('card' in classes and 'restaurant' in classes):
                self.in_restaurant_card = True
                self.current_restaurant = {}
            
            if self.in_restaurant_card:
                if 'restaurant-name' in classes or ('text' in classes and 'font-semibold' in classes):
                    self.in_name = True
                    self.capture_text = True
                elif 'price-range' in classes or 'text-gray' in classes:
                    self.in_price = True
                    self.capture_text = True
    
    def handle_data(self, data):
        text = data.strip()
        if self.capture_text and text:
            self.text_buffer.append(text)
            
            if self.in_select:
                if 'Price: Low to High' in text:
                    self.sort_value = 'price_low_high'
                elif 'Highest Rated' in text:
                    self.sort_value = 'highest_rated'
    
    def handle_endtag(self, tag):
        if tag == 'select':
            self.in_select = False
            
        if self.in_name:
            if self.text_buffer:
                self.current_restaurant['name'] = ' '.join(self.text_buffer)
                self.text_buffer = []
            self.in_name = False
            self.capture_text = False
            
        if self.in_price:
            if self.text_buffer:
                self.current_restaurant['price'] = ' '.join(self.text_buffer)
                self.text_buffer = []
            self.in_price = False
            self.capture_text = False
            
        if self.in_restaurant_card and tag == 'div':
            if self.current_restaurant and 'name' in self.current_restaurant:
                self.restaurants.append(self.current_restaurant.copy())
                self.current_restaurant = {}
            self.in_restaurant_card = False


class RestaurantDetailParser(HTMLParser):
    """Parse restaurant detail page."""
    
    def __init__(self):
        super().__init__()
        self.restaurant_name = None
        self.has_menu_section = False
        self.in_title = False
        self.in_menu = False
        self.capture_text = False
        self.text_buffer = []
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if tag == 'h1' or tag == 'h2':
            self.in_title = True
            self.capture_text = True
        
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            if 'menu' in classes.lower() or 'menu-section' in classes:
                self.has_menu_section = True
                self.in_menu = True
        
        # Check for menu text
        if tag == 'button' or tag == 'a':
            if 'class' in attrs_dict:
                classes = attrs_dict['class']
                if 'menu' in classes.lower():
                    self.has_menu_section = True
    
    def handle_data(self, data):
        text = data.strip()
        if text:
            if 'menu' in text.lower() or 'Menu' in text:
                self.has_menu_section = True
            
            if self.capture_text:
                self.text_buffer.append(text)
    
    def handle_endtag(self, tag):
        if self.in_title and (tag == 'h1' or tag == 'h2'):
            if self.text_buffer and not self.restaurant_name:
                self.restaurant_name = ' '.join(self.text_buffer)
            self.text_buffer = []
            self.in_title = False
            self.capture_text = False


class FavoritesParser(HTMLParser):
    """Parse favorites page to count restaurants."""
    
    def __init__(self):
        super().__init__()
        self.favorites_count = 0
        self.in_favorites = False
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        
        if 'class' in attrs_dict:
            classes = attrs_dict['class']
            # Count restaurant cards in favorites
            if 'restaurant-card' in classes or 'favorite-card' in classes:
                self.favorites_count += 1


class Evaluator:
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_file = self.result_dir / "result.json"
        self.traj_file = self.result_dir / "traj.jsonl"

        # Define checkpoints based on task requirements
        self.checkpoints = {
            "cp1_navigated_to_profile": False,
            "cp2_dietary_preferences_set": False,
            "cp3_navigated_to_discover": False,
            "cp4_sorted_by_price_low_high": False,
            "cp5_viewed_second_restaurant_detail": False,
            "cp6_checked_menu": False,
            "cp7_navigated_to_favorites": False,
            "cp8_removed_first_restaurant": False,
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
        """Checkpoint 1: Navigated to profile page"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Check if profile link was clicked in early steps
            for action in actions[:10]:
                action_data = action.get('action', {})
                if action_data.get('action_type') == 'CLICK':
                    selector = action_data.get('parameters', {}).get('selector', '')
                    if 'Profile' in selector or 'profile' in selector.lower():
                        result = action.get('result', {})
                        if result.get('success'):
                            self.details['profile_navigation'] = 'Success'
                            return True
            
            self.issues.append("Did not navigate to profile page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking profile navigation: {str(e)}")
            return False

    def check_checkpoint_2(self) -> bool:
        """Checkpoint 2: Dietary preferences set to Vegetarian and Dairy-Free"""
        try:
            # Check profile page steps (steps 1-97 show profile page attempts)
            profile_steps = self.find_step_html("step_[1-9]_*")
            profile_steps.extend(self.find_step_html("step_[1-9][0-9]_*"))
            
            # Check final state or late steps for dietary preferences
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found to check dietary preferences")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = DietaryPreferencesParser()
            parser.feed(html_content)
            
            preferences = parser.dietary_preferences
            self.details['dietary_preferences_found'] = preferences
            
            # Check if both Vegetarian and Dairy-Free are set
            has_vegetarian = 'Vegetarian' in preferences
            has_dairy_free = 'Dairy-Free' in preferences
            
            if has_vegetarian and has_dairy_free:
                self.details['dietary_preferences'] = 'Vegetarian and Dairy-Free set correctly'
                return True
            else:
                missing = []
                if not has_vegetarian:
                    missing.append('Vegetarian')
                if not has_dairy_free:
                    missing.append('Dairy-Free')
                self.issues.append(f"Dietary preferences not set correctly. Missing: {', '.join(missing)}")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking dietary preferences: {str(e)}")
            return False

    def check_checkpoint_3(self) -> bool:
        """Checkpoint 3: Navigated to Discover Restaurants page"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Look for navigation to Search/Discover page (step 97)
            for action in actions[90:]:
                action_data = action.get('action', {})
                if action_data.get('action_type') == 'CLICK':
                    selector = action_data.get('parameters', {}).get('selector', '')
                    if 'Search' in selector or 'search' in selector.lower():
                        result = action.get('result', {})
                        if result.get('success'):
                            self.details['discover_navigation'] = 'Success'
                            return True
            
            self.issues.append("Did not navigate to Discover Restaurants page")
            return False
        except Exception as e:
            self.issues.append(f"Error checking discover navigation: {str(e)}")
            return False

    def check_checkpoint_4(self) -> bool:
        """Checkpoint 4: Sorted restaurants by Price: Low to High"""
        try:
            final_html = self.find_final_html()
            if not final_html:
                self.issues.append("No final HTML found")
                return False
            
            with open(final_html, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            parser = RestaurantListParser()
            parser.feed(html_content)
            
            # Check if sort is set to Price: Low to High
            if parser.sort_value == 'price_low_high':
                self.details['sort_option'] = 'Price: Low to High'
                return True
            else:
                self.details['sort_option'] = parser.sort_value or 'Not set'
                self.issues.append(f"Restaurants not sorted by Price: Low to High (current: {self.details['sort_option']})")
                return False
                
        except Exception as e:
            self.issues.append(f"Error checking sort option: {str(e)}")
            return False

    def check_checkpoint_5(self) -> bool:
        """Checkpoint 5: Viewed second restaurant's detail"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # After sorting, need to check if second restaurant was clicked
            # This would happen after step 99 (last recorded step was clicking sort)
            # Since agent hit max_steps at 100, this was not completed
            
            # Check trajectory for restaurant detail page visits
            trajectory = self.get_trajectory()
            
            # Look for navigation to restaurant detail after sorting
            for i, action in enumerate(actions[98:], 98):
                action_data = action.get('action', {})
                reasoning = action_data.get('reasoning', '')
                
                # Check if viewing restaurant detail
                if 'restaurant' in reasoning.lower() and 'detail' in reasoning.lower():
                    self.details['restaurant_detail_viewed'] = 'Yes'
                    return True
            
            self.issues.append("Did not view second restaurant's detail page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking restaurant detail view: {str(e)}")
            return False

    def check_checkpoint_6(self) -> bool:
        """Checkpoint 6: Checked restaurant menu"""
        try:
            # Look for menu-related steps
            step_files = sorted(self.result_dir.glob("step_*_raw.html"))
            
            for step_file in step_files[-10:]:  # Check last 10 steps
                with open(step_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                parser = RestaurantDetailParser()
                parser.feed(html_content)
                
                if parser.has_menu_section:
                    self.details['menu_checked'] = 'Yes'
                    return True
            
            self.issues.append("Did not check restaurant menu")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking menu: {str(e)}")
            return False

    def check_checkpoint_7(self) -> bool:
        """Checkpoint 7: Navigated to favorites page"""
        try:
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Check if favorites link was clicked
            for action in actions:
                action_data = action.get('action', {})
                if action_data.get('action_type') == 'CLICK':
                    selector = action_data.get('parameters', {}).get('selector', '')
                    if 'Favorites' in selector or 'favorites' in selector.lower():
                        result = action.get('result', {})
                        if result.get('success'):
                            self.details['favorites_navigation'] = 'Success'
                            return True
            
            self.issues.append("Did not navigate to favorites page")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking favorites navigation: {str(e)}")
            return False

    def check_checkpoint_8(self) -> bool:
        """Checkpoint 8: Removed first restaurant from favorites"""
        try:
            # This would require checking intermediate steps for the remove action
            result_data = self.load_result_json()
            actions = result_data.get('action_history', [])
            
            # Look for remove button clicks in favorites page
            for action in actions:
                action_data = action.get('action', {})
                reasoning = action_data.get('reasoning', '')
                
                if 'remove' in reasoning.lower() and 'favorite' in reasoning.lower():
                    result = action.get('result', {})
                    if result.get('success'):
                        self.details['removed_from_favorites'] = 'Yes'
                        return True
            
            self.issues.append("Did not remove first restaurant from favorites")
            return False
            
        except Exception as e:
            self.issues.append(f"Error checking favorites removal: {str(e)}")
            return False

    def evaluate(self) -> Dict:
        """Execute complete evaluation."""
        try:
            result_data = self.load_result_json()

            # Run checkpoint validations
            self.checkpoints['cp1_navigated_to_profile'] = self.check_checkpoint_1()
            self.checkpoints['cp2_dietary_preferences_set'] = self.check_checkpoint_2()
            self.checkpoints['cp3_navigated_to_discover'] = self.check_checkpoint_3()
            self.checkpoints['cp4_sorted_by_price_low_high'] = self.check_checkpoint_4()
            self.checkpoints['cp5_viewed_second_restaurant_detail'] = self.check_checkpoint_5()
            self.checkpoints['cp6_checked_menu'] = self.check_checkpoint_6()
            self.checkpoints['cp7_navigated_to_favorites'] = self.check_checkpoint_7()
            self.checkpoints['cp8_removed_first_restaurant'] = self.check_checkpoint_8()

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
                'total_steps': result_data.get('steps', 0)
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
        print("Usage: python reservation_query18.py <result_directory>")
        sys.exit(1)

    result_dir = sys.argv[1]
    evaluator = Evaluator(result_dir)
    evaluation_result = evaluator.evaluate()

    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
    sys.exit(0 if evaluation_result['overall_success'] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Direct client interface for online OSRS.
This module handles direct interaction with the live OSRS game client
using screen capture for state detection and input simulation for actions.
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import threading
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# For screen capture
import cv2
import pyautogui

# For input simulation 
import pyautogui
import keyboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OsrsDirectClient:
    """
    Direct client interface for online OSRS.
    Uses screen capture for game state detection and input simulation for actions.
    """
    
    # Game client window coordinates (to be configured by the user)
    DEFAULT_CLIENT_REGION = (0, 0, 765, 503)  # Default OSRS client dimensions
    
    # Templates directory for image recognition
    TEMPLATES_DIR = "templates"
    
    def __init__(self, client_region=None, templates_dir=None):
        """
        Initialize the OSRS direct client interface.
        
        Args:
            client_region: Tuple of (x, y, width, height) defining the game client region on screen
            templates_dir: Directory containing template images for recognition
        """
        self.client_region = client_region or self.DEFAULT_CLIENT_REGION
        self.templates_dir = templates_dir or self.TEMPLATES_DIR
        
        # Create templates directory if it doesn't exist
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Load template images for recognition
        self.templates = self._load_templates()
        
        # State variables
        self.running = False
        self.current_state = {}
        self.state_lock = threading.Lock()
        
        # Action mapping
        self.action_mapping = {
            0: self._action_eat_food,
            1: self._action_drink_potion,
            2: self._action_switch_prayer,
            3: self._action_switch_weapon,
            4: self._action_special_attack,
            5: self._action_move_character,
            6: self._action_click_opponent,
            7: self._action_overhead_prayer,
            8: self._action_protection_prayer,
            9: self._action_inventory_item,
        }
        
        logger.info("OSRS Direct Client initialized")
    
    def start(self):
        """
        Start the OSRS direct client interface.
        This begins the state detection loop.
        """
        if self.running:
            logger.warning("Client is already running")
            return
        
        self.running = True
        
        # Start state detection in a separate thread
        self.state_thread = threading.Thread(target=self._state_detection_loop, daemon=True)
        self.state_thread.start()
        
        logger.info("OSRS Direct Client started")
    
    def stop(self):
        """Stop the OSRS direct client interface."""
        self.running = False
        if hasattr(self, 'state_thread'):
            self.state_thread.join(timeout=1.0)
        logger.info("OSRS Direct Client stopped")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current game state.
        
        Returns:
            Dict: Current game state
        """
        with self.state_lock:
            return self.current_state.copy()
    
    def perform_action(self, action_id: int) -> bool:
        """
        Perform an action in the game.
        
        Args:
            action_id: ID of the action to perform
            
        Returns:
            bool: True if action was performed successfully
        """
        if not self.running:
            logger.error("Client is not running")
            return False
        
        if action_id not in self.action_mapping:
            logger.error(f"Unknown action ID: {action_id}")
            return False
        
        try:
            # Get the action function
            action_fn = self.action_mapping[action_id]
            
            # Perform the action
            return action_fn()
        
        except Exception as e:
            logger.error(f"Error performing action {action_id}: {e}")
            return False
    
    def _state_detection_loop(self):
        """Background loop for detecting the game state."""
        try:
            while self.running:
                # Capture the game client screen
                screenshot = self._capture_screen()
                
                if screenshot is None:
                    logger.warning("Failed to capture screen")
                    time.sleep(1.0)
                    continue
                
                # Process the screenshot to detect game state
                new_state = self._process_screen(screenshot)
                
                # Update current state
                with self.state_lock:
                    self.current_state = new_state
                
                # Sleep for a short time to avoid high CPU usage
                time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error in state detection loop: {e}")
            self.running = False
    
    def _capture_screen(self) -> Optional[np.ndarray]:
        """
        Capture the game client screen.
        
        Returns:
            np.ndarray: Screenshot as a numpy array, or None if failed
        """
        try:
            # Capture the specified region of the screen
            x, y, width, height = self.client_region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            
            # Convert to numpy array for OpenCV processing
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            return None
    
    def _process_screen(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Process the screenshot to detect game state.
        
        Args:
            screenshot: Screenshot as a numpy array
            
        Returns:
            Dict: Detected game state
        """
        # This is where we would use computer vision to extract game state
        # For example, detecting health, prayer, inventory, etc.
        
        # For now, we'll just create a placeholder state with dummy values
        state = {
            "timestamp": datetime.now().isoformat(),
            "health": self._detect_health(screenshot),
            "prayer": self._detect_prayer(screenshot),
            "spec_energy": self._detect_spec_energy(screenshot),
            "equipped_weapon": self._detect_equipped_weapon(screenshot),
            "inventory": self._detect_inventory(screenshot),
            "player_location": self._detect_player_location(screenshot),
            "opponent_location": self._detect_opponent_location(screenshot),
            "active_prayers": self._detect_active_prayers(screenshot),
        }
        
        return state
    
    def _load_templates(self) -> Dict[str, np.ndarray]:
        """
        Load template images for recognition.
        
        Returns:
            Dict: Dictionary of template images
        """
        templates = {}
        
        # Check if templates directory exists
        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return templates
        
        # Load all template images
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    template_path = os.path.join(self.templates_dir, filename)
                    template_name = os.path.splitext(filename)[0]
                    template_img = cv2.imread(template_path)
                    
                    if template_img is not None:
                        templates[template_name] = template_img
                        logger.debug(f"Loaded template: {template_name}")
                    else:
                        logger.warning(f"Failed to load template: {template_path}")
                
                except Exception as e:
                    logger.error(f"Error loading template {filename}: {e}")
        
        logger.info(f"Loaded {len(templates)} template images")
        return templates
    
    def _detect_template(self, screenshot: np.ndarray, template_name: str, threshold: float = 0.8) -> Tuple[bool, Tuple[int, int]]:
        """
        Detect a template in the screenshot.
        
        Args:
            screenshot: Screenshot as a numpy array
            template_name: Name of the template to detect
            threshold: Matching threshold (0.0 to 1.0)
            
        Returns:
            Tuple of (detected, (x, y)) where detected is True if template was found
            and (x, y) is the center position of the detected template
        """
        if template_name not in self.templates:
            logger.warning(f"Template not found: {template_name}")
            return False, (0, 0)
        
        template = self.templates[template_name]
        
        # Perform template matching
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            # Template found, calculate center position
            template_h, template_w = template.shape[:2]
            center_x = max_loc[0] + template_w // 2
            center_y = max_loc[1] + template_h // 2
            return True, (center_x, center_y)
        
        # Template not found
        return False, (0, 0)
    
    # State detection methods
    
    def _detect_health(self, screenshot: np.ndarray) -> int:
        """
        Detect player health from screenshot.
        
        Args:
            screenshot: Screenshot as a numpy array
            
        Returns:
            int: Detected health value
        """
        # This would use image processing to detect the health bar/value
        # For now, return a placeholder value
        return 99
    
    def _detect_prayer(self, screenshot: np.ndarray) -> int:
        """
        Detect player prayer points from screenshot.
        
        Args:
            screenshot: Screenshot as a numpy array
            
        Returns:
            int: Detected prayer points
        """
        # This would use image processing to detect the prayer bar/value
        # For now, return a placeholder value
        return 99
    
    def _detect_spec_energy(self, screenshot: np.ndarray) -> int:
        """
        Detect special attack energy from screenshot.
        
        Args:
            screenshot: Screenshot as a numpy array
            
        Returns:
            int: Detected special attack energy (0-100)
        """
        # This would use image processing to detect the special attack bar/value
        # For now, return a placeholder value
        return 100
    
    def _detect_equipped_weapon(self, screenshot: np.ndarray) -> str:
        """
        Detect equipped weapon from screenshot.
        
        Args:
            screenshot: Screenshot as a numpy array
            
        Returns:
            str: Name of the equipped weapon
        """
        # This would use image processing to detect the equipped weapon
        # For now, return a placeholder value
        return "Dragon Claws"
    
    def _detect_inventory(self, screenshot: np.ndarray) -> List[str]:
        """
        Detect inventory contents from screenshot.
        
        Args:
            screenshot: Screenshot as a numpy array
            
        Returns:
            List[str]: List of inventory item names
        """
        # This would use image processing to detect the inventory contents
        # For now, return placeholder values
        return [
            "Super Combat", "Anglerfish", "Anglerfish", "Anglerfish",
            "Anglerfish", "Super Restore", "Super Restore", "Super Restore",
            "Shark", "Shark", "Shark", "Shark",
            "Shark", "Shark", "Shark", "Shark",
            "Shark", "Shark", "Shark", "Shark",
            "Shark", "Shark", "Shark", "Shark",
            "Shark", "Shark", "Shark", "Shark"
        ]
    
    def _detect_player_location(self, screenshot: np.ndarray) -> Tuple[int, int]:
        """
        Detect player location from screenshot.
        
        Args:
            screenshot: Screenshot as a numpy array
            
        Returns:
            Tuple[int, int]: (x, y) coordinates of the player
        """
        # This would use image processing to detect the player location
        # For now, return placeholder values
        return (self.client_region[2] // 2, self.client_region[3] // 2)
    
    def _detect_opponent_location(self, screenshot: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect opponent location from screenshot.
        
        Args:
            screenshot: Screenshot as a numpy array
            
        Returns:
            Optional[Tuple[int, int]]: (x, y) coordinates of the opponent, or None if not found
        """
        # This would use image processing to detect the opponent location
        # For now, return placeholder values
        return (self.client_region[2] // 2, self.client_region[3] // 2 - 100)
    
    def _detect_active_prayers(self, screenshot: np.ndarray) -> List[str]:
        """
        Detect active prayers from screenshot.
        
        Args:
            screenshot: Screenshot as a numpy array
            
        Returns:
            List[str]: List of active prayer names
        """
        # This would use image processing to detect active prayers
        # For now, return placeholder values
        return ["Protect from Melee", "Piety"]
    
    # Action methods
    
    def _action_eat_food(self) -> bool:
        """
        Perform the eat food action.
        
        Returns:
            bool: True if action was successful
        """
        logger.info("Performing action: Eat food")
        
        try:
            # Find food in inventory
            food_items = ["Shark", "Anglerfish", "Manta ray"]
            inventory = self.current_state.get("inventory", [])
            
            food_found = False
            for i, item in enumerate(inventory):
                if item in food_items:
                    # Calculate inventory slot position
                    slot_x, slot_y = self._get_inventory_slot_position(i)
                    
                    # Click on the food item
                    pyautogui.click(slot_x, slot_y)
                    food_found = True
                    break
            
            if not food_found:
                logger.warning("No food found in inventory")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error in eat food action: {e}")
            return False
    
    def _action_drink_potion(self) -> bool:
        """
        Perform the drink potion action.
        
        Returns:
            bool: True if action was successful
        """
        logger.info("Performing action: Drink potion")
        
        try:
            # Find potion in inventory
            potion_items = ["Super Combat", "Super Restore", "Saradomin brew"]
            inventory = self.current_state.get("inventory", [])
            
            potion_found = False
            for i, item in enumerate(inventory):
                if any(potion in item for potion in potion_items):
                    # Calculate inventory slot position
                    slot_x, slot_y = self._get_inventory_slot_position(i)
                    
                    # Click on the potion item
                    pyautogui.click(slot_x, slot_y)
                    potion_found = True
                    break
            
            if not potion_found:
                logger.warning("No potion found in inventory")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error in drink potion action: {e}")
            return False
    
    def _action_switch_prayer(self) -> bool:
        """
        Perform the switch prayer action.
        
        Returns:
            bool: True if action was successful
        """
        logger.info("Performing action: Switch prayer")
        
        try:
            # Open prayer tab if needed
            self._ensure_tab_open("prayer")
            
            # Toggle prayer
            prayers = ["Protect from Melee", "Protect from Magic", "Protect from Missiles"]
            prayer_to_activate = random.choice(prayers)
            
            # This would need template matching to find the prayer icon
            # For now, just simulate a click at a random position in the prayer tab area
            prayer_tab_x = self.client_region[0] + self.client_region[2] - 150
            prayer_tab_y = self.client_region[1] + 250
            
            pyautogui.click(prayer_tab_x, prayer_tab_y)
            logger.info(f"Activated prayer: {prayer_to_activate}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error in switch prayer action: {e}")
            return False
    
    def _action_switch_weapon(self) -> bool:
        """
        Perform the switch weapon action.
        
        Returns:
            bool: True if action was successful
        """
        logger.info("Performing action: Switch weapon")
        
        try:
            # Open inventory tab if needed
            self._ensure_tab_open("inventory")
            
            # Find weapon in inventory
            weapon_items = ["Dragon Claws", "Armadyl Godsword", "Abyssal Whip"]
            inventory = self.current_state.get("inventory", [])
            
            weapon_found = False
            for i, item in enumerate(inventory):
                if item in weapon_items:
                    # Calculate inventory slot position
                    slot_x, slot_y = self._get_inventory_slot_position(i)
                    
                    # Click on the weapon item
                    pyautogui.click(slot_x, slot_y)
                    weapon_found = True
                    break
            
            if not weapon_found:
                logger.warning("No weapon found in inventory")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error in switch weapon action: {e}")
            return False
    
    def _action_special_attack(self) -> bool:
        """
        Perform the special attack action.
        
        Returns:
            bool: True if action was successful
        """
        logger.info("Performing action: Special attack")
        
        try:
            # Check if we have enough special attack energy
            spec_energy = self.current_state.get("spec_energy", 0)
            if spec_energy < 25:
                logger.warning("Not enough special attack energy")
                return False
            
            # Click on the special attack bar
            spec_bar_x = self.client_region[0] + 560
            spec_bar_y = self.client_region[1] + 145
            
            pyautogui.click(spec_bar_x, spec_bar_y)
            
            return True
        
        except Exception as e:
            logger.error(f"Error in special attack action: {e}")
            return False
    
    def _action_move_character(self) -> bool:
        """
        Perform the move character action.
        
        Returns:
            bool: True if action was successful
        """
        logger.info("Performing action: Move character")
        
        try:
            # Calculate a random position to move to
            center_x = self.client_region[0] + self.client_region[2] // 2
            center_y = self.client_region[1] + self.client_region[3] // 2
            
            offset_x = random.randint(-100, 100)
            offset_y = random.randint(-100, 100)
            
            move_x = center_x + offset_x
            move_y = center_y + offset_y
            
            # Ensure the position is within the client region
            move_x = max(self.client_region[0], min(move_x, self.client_region[0] + self.client_region[2]))
            move_y = max(self.client_region[1], min(move_y, self.client_region[1] + self.client_region[3]))
            
            # Click to move
            pyautogui.click(move_x, move_y)
            
            return True
        
        except Exception as e:
            logger.error(f"Error in move character action: {e}")
            return False
    
    def _action_click_opponent(self) -> bool:
        """
        Perform the click opponent action.
        
        Returns:
            bool: True if action was successful
        """
        logger.info("Performing action: Click opponent")
        
        try:
            # Get opponent location
            opponent_loc = self.current_state.get("opponent_location")
            
            if not opponent_loc:
                logger.warning("Opponent not detected")
                return False
            
            # Click on opponent
            opponent_x, opponent_y = opponent_loc
            pyautogui.click(self.client_region[0] + opponent_x, self.client_region[1] + opponent_y)
            
            return True
        
        except Exception as e:
            logger.error(f"Error in click opponent action: {e}")
            return False
    
    def _action_overhead_prayer(self) -> bool:
        """
        Perform the overhead prayer action.
        
        Returns:
            bool: True if action was successful
        """
        logger.info("Performing action: Overhead prayer")
        
        try:
            # Open prayer tab if needed
            self._ensure_tab_open("prayer")
            
            # Toggle overhead prayer
            overhead_prayers = ["Protect from Melee", "Protect from Magic", "Protect from Missiles"]
            prayer_to_activate = random.choice(overhead_prayers)
            
            # This would need template matching to find the prayer icon
            # For now, just simulate a click at a position in the prayer tab area
            prayer_tab_x = self.client_region[0] + self.client_region[2] - 150
            prayer_tab_y = self.client_region[1] + 200
            
            pyautogui.click(prayer_tab_x, prayer_tab_y)
            logger.info(f"Activated overhead prayer: {prayer_to_activate}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error in overhead prayer action: {e}")
            return False
    
    def _action_protection_prayer(self) -> bool:
        """
        Perform the protection prayer action.
        
        Returns:
            bool: True if action was successful
        """
        logger.info("Performing action: Protection prayer")
        
        try:
            # Open prayer tab if needed
            self._ensure_tab_open("prayer")
            
            # Toggle protection prayer
            protection_prayers = ["Piety", "Rigour", "Augury"]
            prayer_to_activate = random.choice(protection_prayers)
            
            # This would need template matching to find the prayer icon
            # For now, just simulate a click at a position in the prayer tab area
            prayer_tab_x = self.client_region[0] + self.client_region[2] - 75
            prayer_tab_y = self.client_region[1] + 300
            
            pyautogui.click(prayer_tab_x, prayer_tab_y)
            logger.info(f"Activated protection prayer: {prayer_to_activate}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error in protection prayer action: {e}")
            return False
    
    def _action_inventory_item(self) -> bool:
        """
        Perform the click inventory item action.
        
        Returns:
            bool: True if action was successful
        """
        logger.info("Performing action: Click inventory item")
        
        try:
            # Open inventory tab if needed
            self._ensure_tab_open("inventory")
            
            # Click on a random inventory slot
            inventory = self.current_state.get("inventory", [])
            if not inventory:
                logger.warning("No items in inventory")
                return False
            
            # Select a random item
            slot_index = random.randint(0, min(27, len(inventory) - 1))
            
            # Calculate inventory slot position
            slot_x, slot_y = self._get_inventory_slot_position(slot_index)
            
            # Click on the item
            pyautogui.click(slot_x, slot_y)
            
            return True
        
        except Exception as e:
            logger.error(f"Error in inventory item action: {e}")
            return False
    
    # Helper methods
    
    def _ensure_tab_open(self, tab_name: str):
        """
        Ensure the specified tab is open.
        
        Args:
            tab_name: Name of the tab to open (e.g., "inventory", "prayer")
        """
        # Define tab positions
        tab_positions = {
            "inventory": (626, 170),
            "prayer": (626, 205),
            "equipment": (626, 240),
            "magic": (626, 275),
        }
        
        if tab_name in tab_positions:
            tab_x, tab_y = tab_positions[tab_name]
            tab_x += self.client_region[0]
            tab_y += self.client_region[1]
            
            # Click on the tab
            pyautogui.click(tab_x, tab_y)
            time.sleep(0.1)
    
    def _get_inventory_slot_position(self, slot_index: int) -> Tuple[int, int]:
        """
        Calculate the screen position of an inventory slot.
        
        Args:
            slot_index: Index of the inventory slot (0-27)
            
        Returns:
            Tuple[int, int]: (x, y) coordinates of the inventory slot
        """
        # Define inventory grid
        inventory_start_x = self.client_region[0] + 560
        inventory_start_y = self.client_region[1] + 210
        slot_width = 42
        slot_height = 36
        
        # Calculate row and column
        row = slot_index // 4
        col = slot_index % 4
        
        # Calculate slot position
        slot_x = inventory_start_x + col * slot_width + slot_width // 2
        slot_y = inventory_start_y + row * slot_height + slot_height // 2
        
        return (slot_x, slot_y)


def calibrate_client_region():
    """
    Interactive procedure to calibrate the client region.
    
    Returns:
        Tuple[int, int, int, int]: Client region as (x, y, width, height)
    """
    print("\n=== OSRS Client Region Calibration ===")
    print("Please follow these steps to calibrate the client region:")
    print("1. Make sure your OSRS client is visible on screen")
    print("2. Position your mouse on the TOP-LEFT corner of the client area")
    print("3. Press ENTER when ready")
    input()
    
    top_left = pyautogui.position()
    print(f"Top-left corner: {top_left}")
    
    print("\n4. Now position your mouse on the BOTTOM-RIGHT corner of the client area")
    print("5. Press ENTER when ready")
    input()
    
    bottom_right = pyautogui.position()
    print(f"Bottom-right corner: {bottom_right}")
    
    # Calculate client region
    x = top_left[0]
    y = top_left[1]
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    
    client_region = (x, y, width, height)
    print(f"\nDetected client region: {client_region}")
    
    return client_region


def capture_templates(client_region, templates_dir):
    """
    Interactive procedure to capture template images for recognition.
    
    Args:
        client_region: Client region as (x, y, width, height)
        templates_dir: Directory to save templates
    """
    print("\n=== Template Image Capture ===")
    print("This procedure will help you capture template images for recognition.")
    print("For each template, follow the instructions to capture the image.")
    
    # Create templates directory if it doesn't exist
    os.makedirs(templates_dir, exist_ok=True)
    
    templates_to_capture = [
        ("health_bar", "the health bar"),
        ("prayer_bar", "the prayer bar"),
        ("spec_bar", "the special attack bar"),
        ("inventory_tab", "the inventory tab icon"),
        ("prayer_tab", "the prayer tab icon"),
        ("protect_melee", "the Protect from Melee prayer icon"),
        ("protect_magic", "the Protect from Magic prayer icon"),
        ("protect_missiles", "the Protect from Missiles prayer icon"),
        ("piety", "the Piety prayer icon"),
        ("weapon_dclaws", "a Dragon Claws in inventory"),
        ("weapon_ags", "an Armadyl Godsword in inventory"),
        ("food_shark", "a Shark in inventory"),
    ]
    
    for template_name, description in templates_to_capture:
        print(f"\nCapturing template for: {description}")
        print(f"1. Position your mouse over {description}")
        print("2. Press ENTER when ready (or type 'skip' to skip this template)")
        
        user_input = input()
        if user_input.lower() == "skip":
            print(f"Skipped {template_name}")
            continue
        
        # Get mouse position
        pos = pyautogui.position()
        
        # Define capture region (small area around mouse position)
        region_size = 40
        capture_region = (
            max(0, pos[0] - region_size // 2),
            max(0, pos[1] - region_size // 2),
            region_size,
            region_size
        )
        
        # Capture screenshot of the region
        screenshot = pyautogui.screenshot(region=capture_region)
        template_path = os.path.join(templates_dir, f"{template_name}.png")
        screenshot.save(template_path)
        
        print(f"Captured template: {template_path}")
    
    print("\nTemplate capture complete!")


def main():
    """Main entry point for the OSRS direct client."""
    parser = argparse.ArgumentParser(description="OSRS Direct Client")
    
    parser.add_argument("--calibrate", action="store_true", help="Calibrate client region")
    parser.add_argument("--capture-templates", action="store_true", help="Capture template images")
    parser.add_argument("--templates-dir", default="templates", help="Templates directory")
    parser.add_argument("--client-x", type=int, help="Client region X coordinate")
    parser.add_argument("--client-y", type=int, help="Client region Y coordinate")
    parser.add_argument("--client-width", type=int, help="Client region width")
    parser.add_argument("--client-height", type=int, help="Client region height")
    
    args = parser.parse_args()
    
    # Determine client region
    client_region = None
    
    if args.calibrate:
        client_region = calibrate_client_region()
    elif args.client_x is not None and args.client_y is not None and \
         args.client_width is not None and args.client_height is not None:
        client_region = (args.client_x, args.client_y, args.client_width, args.client_height)
    
    # Capture templates if requested
    if args.capture_templates:
        if client_region is None:
            print("Error: Client region not specified. Use --calibrate or provide region coordinates.")
            return 1
        
        capture_templates(client_region, args.templates_dir)
        return 0
    
    # Create and start the client
    client = OsrsDirectClient(client_region, args.templates_dir)
    client.start()
    
    try:
        print("OSRS Direct Client started. Press Ctrl+C to stop.")
        print("\nAvailable actions:")
        for action_id, action_fn in client.action_mapping.items():
            print(f"{action_id}: {action_fn.__name__.replace('_action_', '')}")
        
        print("\nEnter action ID to perform (or 'q' to quit):")
        
        while True:
            user_input = input("> ")
            
            if user_input.lower() == 'q':
                break
            
            try:
                action_id = int(user_input)
                success = client.perform_action(action_id)
                print(f"Action {'succeeded' if success else 'failed'}")
            except ValueError:
                print("Invalid input. Enter an action ID or 'q' to quit.")
    
    except KeyboardInterrupt:
        print("\nStopping client...")
    
    finally:
        client.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
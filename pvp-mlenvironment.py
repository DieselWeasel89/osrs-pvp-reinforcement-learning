"""
Environment wrapper for OSRS PvP reinforcement learning.
Handles communication with the simulation RSPS.
"""

import os
import json
import logging
import socket
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OsrsPvPEnvironment:
    """
    Environment class for interacting with the OSRS PvP simulation.
    """
    
    def __init__(self, env_name: str, server_host: str = "localhost", server_port: int = 43594):
        """
        Initialize the environment.
        
        Args:
            env_name: Name of the environment (must correspond to a contract file)
            server_host: Host of the simulation server
            server_port: Port of the simulation server
        """
        self.env_name = env_name
        self.server_host = server_host
        self.server_port = server_port
        
        # Load environment contract
        contract_path = Path('../contracts/environments') / f"{env_name}.json"
        if not contract_path.exists():
            raise ValueError(f"Environment contract not found: {contract_path}")
        
        with open(contract_path, 'r') as f:
            self.contract = json.load(f)
        
        # Extract information from contract
        self.action_space = self.contract["actions"]
        self.observation_space = self.contract["observations"]
        
        # Initialize socket connection
        self.socket = None
        self.connected = False
        
        # Environment state
        self.current_state = None
        self.episode_step = 0
        self.total_reward = 0
        
        logger.info(f"Initialized environment {env_name} with {len(self.action_space)} actions and {len(self.observation_space)} observations")
    
    def connect(self) -> bool:
        """
        Connect to the simulation server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.connected:
            return True
        
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            
            # Send environment setup message
            setup_msg = {
                "type": "setup",
                "env_name": self.env_name
            }
            self._send_message(setup_msg)
            
            # Receive response
            response = self._receive_message()
            if response.get("status") == "ok":
                self.connected = True
                logger.info(f"Connected to simulation server at {self.server_host}:{self.server_port}")
                return True
            else:
                error = response.get("error", "Unknown error")
                logger.error(f"Failed to setup environment: {error}")
                self.socket.close()
                return False
        
        except Exception as e:
            logger.exception(f"Failed to connect to simulation server: {e}")
            if self.socket:
                self.socket.close()
            return False
    
    def disconnect(self):
        """Disconnect from the simulation server."""
        if self.connected and self.socket:
            try:
                # Send disconnect message
                self._send_message({"type": "disconnect"})
                self.socket.close()
            except:
                pass
            
            self.socket = None
            self.connected = False
            logger.info("Disconnected from simulation server")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and start a new episode.
        
        Returns:
            np.ndarray: Initial observation
        """
        if not self.connected:
            if not self.connect():
                raise ConnectionError("Cannot reset: not connected to simulation server")
        
        # Send reset message
        self._send_message({"type": "reset"})
        
        # Receive initial state
        response = self._receive_message()
        if response.get("type") != "state":
            error = response.get("error", "Unknown error")
            raise RuntimeError(f"Failed to reset environment: {error}")
        
        # Extract observation
        self.current_state = response.get("state", {})
        observation = self._extract_observation(self.current_state)
        
        # Reset environment state
        self.episode_step = 0
        self.total_reward = 0
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (index into action space)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not self.connected:
            raise ConnectionError("Cannot step: not connected to simulation server")
        
        if action < 0 or action >= len(self.action_space):
            raise ValueError(f"Invalid action: {action}. Must be between 0 and {len(self.action_space) - 1}")
        
        # Send action message
        action_name = self.action_space[action]
        self._send_message({
            "type": "action",
            "action": action_name
        })
        
        # Receive new state
        response = self._receive_message()
        if response.get("type") != "state":
            error = response.get("error", "Unknown error")
            raise RuntimeError(f"Failed to take step: {error}")
        
        # Extract observation, reward, done
        self.current_state = response.get("state", {})
        observation = self._extract_observation(self.current_state)
        reward = self.current_state.get("reward", 0.0)
        done = self.current_state.get("done", False)
        
        # Update environment state
        self.episode_step += 1
        self.total_reward += reward
        
        # Additional info
        info = {
            "step": self.episode_step,
            "total_reward": self.total_reward,
            "action_name": action_name
        }
        
        return observation, reward, done, info
    
    def _extract_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Extract observation from state according to the contract.
        
        Args:
            state: Full state from the simulation
            
        Returns:
            np.ndarray: Observation vector
        """
        observation = []
        
        for obs_name in self.observation_space:
            value = state.get(obs_name, 0.0)
            observation.append(float(value))
        
        return np.array(observation, dtype=np.float32)
    
    def _send_message(self, message: Dict[str, Any]):
        """
        Send a message to the simulation server.
        
        Args:
            message: Message to send
        """
        if not self.socket:
            raise ConnectionError("Socket not initialized")
        
        message_json = json.dumps(message)
        self.socket.sendall(message_json.encode('utf-8') + b'\n')
    
    def _receive_message(self) -> Dict[str, Any]:
        """
        Receive a message from the simulation server.
        
        Returns:
            Dict: Received message
        """
        if not self.socket:
            raise ConnectionError("Socket not initialized")
        
        data = b''
        while not data.endswith(b'\n'):
            chunk = self.socket.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed by server")
            data += chunk
        
        message_json = data.decode('utf-8').strip()
        return json.loads(message_json)


class OsrsPvPSelfPlayEnvironment(OsrsPvPEnvironment):
    """
    Extension of the base environment to support self-play training.
    """
    
    def __init__(self, env_name: str, server_host: str = "localhost", server_port: int = 43594):
        """
        Initialize the self-play environment.
        
        Args:
            env_name: Name of the environment (must correspond to a contract file)
            server_host: Host of the simulation server
            server_port: Port of the simulation server
        """
        super().__init__(env_name, server_host, server_port)
        
        # Self-play specific state
        self.opponent_agent = None
        self.is_self_play = False
    
    def set_opponent_agent(self, agent):
        """
        Set the opponent agent for self-play.
        
        Args:
            agent: Agent to use as opponent (None for default opponent)
        """
        self.opponent_agent = agent
        self.is_self_play = (agent is not None)
        
        # Send self-play configuration if connected
        if self.connected:
            self._send_selfplay_config()
    
    def connect(self) -> bool:
        """
        Connect to the simulation server with self-play configuration.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        success = super().connect()
        
        if success and self.is_self_play:
            self._send_selfplay_config()
        
        return success
    
    def _send_selfplay_config(self):
        """Send self-play configuration to the server."""
        self._send_message({
            "type": "selfplay_config",
            "enabled": self.is_self_play
        })
        
        response = self._receive_message()
        if response.get("status") != "ok":
            error = response.get("error", "Unknown error")
            logger.error(f"Failed to set self-play configuration: {error}")
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment, handling opponent actions in self-play.
        
        Args:
            action: Action to take (index into action space)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not self.is_self_play or not self.opponent_agent:
            # Regular step if not in self-play mode
            return super().step(action)
        
        # Send action message with self-play flag
        action_name = self.action_space[action]
        self._send_message({
            "type": "action",
            "action": action_name,
            "selfplay": True
        })
        
        # Receive opponent state
        response = self._receive_message()
        if response.get("type") != "opponent_state":
            error = response.get("error", "Unknown error")
            raise RuntimeError(f"Failed to get opponent state: {error}")
        
        # Get opponent state and action
        opponent_state = response.get("state", {})
        opponent_observation = self._extract_observation(opponent_state)
        opponent_action = self.opponent_agent.get_action(opponent_observation)
        
        # Send opponent action
        opponent_action_name = self.action_space[opponent_action]
        self._send_message({
            "type": "opponent_action",
            "action": opponent_action_name
        })
        
        # Receive new state after both actions
        response = self._receive_message()
        if response.get("type") != "state":
            error = response.get("error", "Unknown error")
            raise RuntimeError(f"Failed to take step: {error}")
        
        # Extract observation, reward, done
        self.current_state = response.get("state", {})
        observation = self._extract_observation(self.current_state)
        reward = self.current_state.get("reward", 0.0)
        done = self.current_state.get("done", False)
        
        # Update environment state
        self.episode_step += 1
        self.total_reward += reward
        
        # Additional info
        info = {
            "step": self.episode_step,
            "total_reward": self.total_reward,
            "action_name": action_name,
            "opponent_action": opponent_action_name
        }
        
        return observation, reward, done, info


# Mock environment for testing without server
class MockOsrsPvPEnvironment(OsrsPvPEnvironment):
    """
    Mock environment for testing without a simulation server.
    """
    
    def __init__(self, env_name: str):
        """
        Initialize the mock environment.
        
        Args:
            env_name: Name of the environment (must correspond to a contract file)
        """
        # Load environment contract
        contract_path = Path('../contracts/environments') / f"{env_name}.json"
        if not contract_path.exists():
            raise ValueError(f"Environment contract not found: {contract_path}")
        
        with open(contract_path, 'r') as f:
            self.contract = json.load(f)
        
        # Extract information from contract
        self.env_name = env_name
        self.action_space = self.contract["actions"]
        self.observation_space = self.contract["observations"]
        
        # Environment state
        self.current_state = {}
        self.episode_step = 0
        self.total_reward = 0
        self.player_health = 99
        self.opponent_health = 99
        self.connected = True
        
        logger.info(f"Initialized mock environment {env_name} with {len(self.action_space)} actions and {len(self.observation_space)} observations")
    
    def connect(self) -> bool:
        """
        Connect to the mock server (always succeeds).
        
        Returns:
            bool: Always True
        """
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect from the mock server."""
        self.connected = False
    
    def reset(self) -> np.ndarray:
        """
        Reset the mock environment.
        
        Returns:
            np.ndarray: Initial observation
        """
        # Reset state
        self.episode_step = 0
        self.total_reward = 0
        self.player_health = 99
        self.opponent_health = 99
        
        # Create mock state
        self._update_state()
        
        # Extract observation
        observation = self._extract_observation(self.current_state)
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the mock environment.
        
        Args:
            action: Action to take (index into action space)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if action < 0 or action >= len(self.action_space):
            raise ValueError(f"Invalid action: {action}. Must be between 0 and {len(self.action_space) - 1}")
        
        # Simulate action effects
        action_name = self.action_space[action]
        
        # Simple simulation: attacking reduces opponent health, eating restores player health
        if "attack" in action_name.lower():
            damage = random.randint(0, 25)
            self.opponent_health = max(0, self.opponent_health - damage)
        elif "eat" in action_name.lower():
            heal = random.randint(5, 20)
            self.player_health = min(99, self.player_health + heal)
        
        # Opponent also takes random action
        opponent_damage = random.randint(0, 15)
        self.player_health = max(0, self.player_health - opponent_damage)
        
        # Update state
        self._update_state()
        
        # Calculate reward
        prev_reward = self.total_reward
        done = self.player_health <= 0 or self.opponent_health <= 0
        
        # Reward is winning (1.0) or damage dealt minus damage taken
        if done:
            if self.opponent_health <= 0:
                reward = 1.0  # Win
            else:
                reward = -1.0  # Loss
        else:
            reward = opponent_damage / 100.0  # Small reward for damage dealt
        
        self.total_reward += reward
        
        # Extract observation
        observation = self._extract_observation(self.current_state)
        
        # Update step counter
        self.episode_step += 1
        
        # Additional info
        info = {
            "step": self.episode_step,
            "total_reward": self.total_reward,
            "action_name": action_name
        }
        
        return observation, reward, done, info
    
    def _update_state(self):
        """Update the mock state based on current values."""
        self.current_state = {
            "player_health": self.player_health / 99.0,
            "opponent_health": self.opponent_health / 99.0,
            "player_prayer": random.random(),
            "opponent_prayer": random.random(),
            "player_special_energy": random.random(),
            "reward": 0.0,
            "done": False
        }
        
        # Add all observations from contract with random values for those not explicitly set
        for obs_name in self.observation_space:
            if obs_name not in self.current_state:
                self.current_state[obs_name] = random.random()

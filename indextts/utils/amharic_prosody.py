"""Amharic Prosody Controls

Handles Amharic-specific phonetic features:
- Gemination (doubled consonants: ሁለት vs ሁሌት)
- Ejective consonants (glottalized: ጥ, ቅ, ጭ, etc.)
- Syllable duration and stress patterns
"""
import torch
import re
from typing import Dict, List


class AmharicProsodyController:
    """Controls Amharic-specific prosodic features"""
    
    # Gemination marker in Ethiopic script
    GEMINATION_MARKER = '፟'
    
    # Ejective consonants (glottalized)
    EJECTIVE_CONSONANTS = [
        'ጠ', 'ጡ', 'ጢ', 'ጣ', 'ጤ', 'ጥ', 'ጦ',  # t'
        'ቀ', 'ቁ', 'ቂ', 'ቃ', 'ቄ', 'ቅ', 'ቆ',  # q'
        'ጨ', 'ጩ', 'ጪ', 'ጫ', 'ጬ', 'ጭ', 'ጮ',  # ch'
        'ጸ', 'ጹ', 'ጺ', 'ጻ', 'ጼ', 'ጽ', 'ጾ',  # ts'
        'ጰ', 'ጱ', 'ጲ', 'ጳ', 'ጴ', 'ጵ', 'ጶ',  # p'
    ]
    
    def __init__(self):
        self.gemination_strength = 1.0
        self.ejective_strength = 1.0
        self.syllable_duration = 1.0
        self.stress_pattern = 'penultimate'  # Typical for Amharic
    
    def apply_gemination_emphasis(self, text: str, strength: float = 1.0) -> str:
        """Apply gemination emphasis
        
        Args:
            text: Amharic text
            strength: Emphasis strength (0.5-2.0)
        
        Returns:
            Text with gemination markers adjusted
        """
        self.gemination_strength = strength
        
        # Detect potential gemination (repeated consonants)
        # In practice, this would interact with phoneme durations
        # For now, we mark potential gemination sites
        
        if strength > 1.2:
            # Strong emphasis: ensure gemination marker present
            # (This is simplified - real implementation would
            # work at the phoneme/duration level)
            pass
        
        return text
    
    def detect_ejectives(self, text: str) -> List[int]:
        """Detect ejective consonant positions
        
        Args:
            text: Amharic text
        
        Returns:
            List of character indices where ejectives occur
        """
        positions = []
        for i, char in enumerate(text):
            if char in self.EJECTIVE_CONSONANTS:
                positions.append(i)
        return positions
    
    def apply_ejective_emphasis(self, text: str, strength: float = 1.0) -> Dict:
        """Apply ejective consonant emphasis
        
        Args:
            text: Amharic text
            strength: Emphasis strength (0.5-2.0)
        
        Returns:
            Dictionary with ejective positions and strength
        """
        self.ejective_strength = strength
        positions = self.detect_ejectives(text)
        
        return {
            'ejective_positions': positions,
            'ejective_strength': strength,
            'num_ejectives': len(positions)
        }
    
    def calculate_syllable_stress(self, text: str, pattern: str = 'penultimate') -> List[float]:
        """Calculate syllable stress pattern
        
        Args:
            text: Amharic text
            pattern: Stress pattern ('penultimate', 'final', 'initial')
        
        Returns:
            List of stress values per syllable (0.0-1.0)
        """
        # Simplified syllable detection
        # In Amharic, each character is typically a syllable (CV structure)
        syllables = [c for c in text if '\u1200' <= c <= '\u137F']
        num_syllables = len(syllables)
        
        stress_values = [0.3] * num_syllables  # Base stress
        
        if num_syllables == 0:
            return []
        
        if pattern == 'penultimate' and num_syllables >= 2:
            # Stress second-to-last syllable (typical Amharic)
            stress_values[-2] = 1.0
        elif pattern == 'final' and num_syllables >= 1:
            # Stress final syllable
            stress_values[-1] = 1.0
        elif pattern == 'initial' and num_syllables >= 1:
            # Stress first syllable
            stress_values[0] = 1.0
        
        return stress_values
    
    def apply_duration_control(self, duration_multiplier: float = 1.0) -> Dict:
        """Apply syllable duration control
        
        Args:
            duration_multiplier: Duration multiplier (0.7-1.3)
        
        Returns:
            Dictionary with duration parameters
        """
        self.syllable_duration = duration_multiplier
        
        return {
            'syllable_duration_multiplier': duration_multiplier,
            'speaking_rate': 1.0 / duration_multiplier  # Inverse relationship
        }
    
    def get_prosody_parameters(self) -> Dict:
        """Get all current prosody parameters
        
        Returns:
            Dictionary of all prosody settings
        """
        return {
            'gemination_strength': self.gemination_strength,
            'ejective_strength': self.ejective_strength,
            'syllable_duration': self.syllable_duration,
            'stress_pattern': self.stress_pattern
        }

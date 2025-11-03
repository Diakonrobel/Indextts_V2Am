import torch
import torchaudio
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd

from indextts.utils.audio_quality_metrics import calculate_audio_quality_metrics


class ModelComparator:
    def __init__(self):
        self.model_a = None
        self.model_b = None
        self.model_a_name = ""
        self.model_b_name = ""
    
    def load_model_a(self, checkpoint_path: str, name: str = "Model A"):
        self.model_a_name = name
        # Model loading would happen here
        return f"Loaded {name} from {checkpoint_path}"
    
    def load_model_b(self, checkpoint_path: str, name: str = "Model B"):
        self.model_b_name = name
        return f"Loaded {name} from {checkpoint_path}"
    
    def compare_outputs(self, audio_a_path: str, audio_b_path: str) -> pd.DataFrame:
        metrics_a = calculate_audio_quality_metrics(audio_a_path)
        metrics_b = calculate_audio_quality_metrics(audio_b_path)
        
        comparison_data = []
        for key in ['rms_energy', 'peak_level', 'quality_score', 'duration_seconds']:
            val_a = metrics_a.get(key, 0)
            val_b = metrics_b.get(key, 0)
            
            if val_a > val_b:
                winner = self.model_a_name
            elif val_b > val_a:
                winner = self.model_b_name
            else:
                winner = "Tie"
            
            comparison_data.append([
                key.replace('_', ' ').title(),
                f"{val_a:.4f}" if isinstance(val_a, float) else val_a,
                f"{val_b:.4f}" if isinstance(val_b, float) else val_b,
                winner
            ])
        
        df = pd.DataFrame(
            comparison_data,
            columns=["Metric", self.model_a_name, self.model_b_name, "Winner"]
        )
        
        return df
    
    def determine_overall_winner(self, comparison_df: pd.DataFrame) -> str:
        winners = comparison_df["Winner"].value_counts().to_dict()
        
        a_wins = winners.get(self.model_a_name, 0)
        b_wins = winners.get(self.model_b_name, 0)
        
        if a_wins > b_wins:
            return f"🏆 {self.model_a_name} wins ({a_wins} metrics)"
        elif b_wins > a_wins:
            return f"🏆 {self.model_b_name} wins ({b_wins} metrics)"
        else:
            return "🤝 Tie - Both models perform equally"

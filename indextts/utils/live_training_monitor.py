import torch
import json
import plotly.graph_objects as go
from pathlib import Path
import re
from datetime import datetime


class LiveTrainingMonitor:
    def __init__(self, log_dir="logs/training"):
        self.log_dir = Path(log_dir)
        self.loss_history = []
        self.val_loss_history = []
        self.steps = []
        self.last_position = 0
        self.metrics = {}
    
    def parse_training_log(self, log_file):
        if not Path(log_file).exists():
            return None
        
        with open(log_file) as f:
            f.seek(self.last_position)
            new_lines = f.readlines()
            self.last_position = f.tell()
        
        for line in new_lines:
            if 'Step' in line and 'Loss' in line:
                match = re.search(r'Step\s+(\d+).*Loss:\s+([\d.]+)', line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    self.steps.append(step)
                    self.loss_history.append(loss)
        
        return self.get_loss_plot()
    
    def get_loss_plot(self):
        if not self.loss_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No training data yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.steps,
            y=self.loss_history,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#E74C3C', width=2)
        ))
        
        fig.update_layout(
            title='Training Loss Curve',
            xaxis_title='Step',
            yaxis_title='Loss',
            template='plotly_white'
        )
        
        return fig
    
    def get_current_metrics(self):
        if not self.loss_history:
            return {}
        
        return {
            'current_step': self.steps[-1] if self.steps else 0,
            'current_loss': self.loss_history[-1] if self.loss_history else 0,
            'min_loss': min(self.loss_history) if self.loss_history else 0,
            'avg_loss': sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0,
            'total_steps': len(self.steps)
        }

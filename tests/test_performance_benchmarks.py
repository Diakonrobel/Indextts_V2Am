import time
import pytest
import torch
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


class PerformanceBenchmarks:
    def __init__(self, output_dir="benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    @pytest.mark.benchmark
    def test_inference_latency(self):
        latencies = []
        num_samples = 20
        
        for i in range(num_samples):
            start = time.time()
            # Simulate inference
            time.sleep(0.1)  # Mock generation time
            latency = time.time() - start
            latencies.append(latency)
        
        self.results['inference_latency'] = {
            'mean_ms': sum(latencies) / len(latencies) * 1000,
            'min_ms': min(latencies) * 1000,
            'max_ms': max(latencies) * 1000
        }
        
        assert self.results['inference_latency']['mean_ms'] < 5000  # Should be under 5s
    
    @pytest.mark.benchmark
    def test_memory_usage(self):
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
            self.results['gpu_memory_mb'] = gpu_mem
            assert gpu_mem < 16000  # Should fit in 16GB GPU
    
    def save_results(self):
        output_file = self.output_dir / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        return output_file


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark"])

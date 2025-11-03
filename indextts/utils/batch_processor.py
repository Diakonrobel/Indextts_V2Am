from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable
import time
from tqdm import tqdm


class BatchTTSProcessor:
    def __init__(self, tts_func: Callable, max_workers: int = 4):
        self.tts_func = tts_func
        self.max_workers = max_workers
    
    def process_batch(self, texts: List[str], voice_id: int = 0, 
                     **generation_kwargs) -> List[Dict]:
        results = []
        
        if self.max_workers == 1:
            # Sequential processing
            for i, text in enumerate(tqdm(texts, desc="Generating")):
                result = self._process_single(i, text, voice_id, **generation_kwargs)
                results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_single, i, text, voice_id, **generation_kwargs
                    ): i
                    for i, text in enumerate(texts)
                }
                
                for future in tqdm(as_completed(futures), total=len(texts), desc="Generating"):
                    result = future.result()
                    results.append(result)
        
        # Sort by index
        results.sort(key=lambda x: x['index'])
        return results
    
    def _process_single(self, index: int, text: str, voice_id: int, **kwargs) -> Dict:
        try:
            start_time = time.time()
            audio_path, status = self.tts_func(text, voice_id, **kwargs)
            duration = time.time() - start_time
            
            return {
                'index': index,
                'text': text,
                'status': 'success',
                'audio_path': audio_path,
                'generation_time': duration,
                'error': None
            }
        except Exception as e:
            return {
                'index': index,
                'text': text,
                'status': 'failed',
                'audio_path': None,
                'generation_time': 0,
                'error': str(e)
            }

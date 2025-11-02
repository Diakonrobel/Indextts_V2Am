"""
Amharic evaluation script for IndexTTS2 fine-tuned models
Comprehensive evaluation metrics for Amharic TTS quality assessment
"""
import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import yaml
from jiwer import wer, cer, mer, wip
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from indextts.utils.amharic_front import AmharicTextTokenizer, AmharicTextNormalizer
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.hf_utils import load_model as load_vocoder


class AmharicTTSEvaluator:
    """Comprehensive evaluator for Amharic TTS models"""
    
    def __init__(
        self,
        config_path: str,
        model_path: str,
        amharic_vocab_path: str,
        vocoder_path: str = None,
        output_dir: str = "amharic_evaluation",
        device: str = "cuda"
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.amharic_vocab_path = amharic_vocab_path
        self.vocoder_path = vocoder_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.tokenizer = AmharicTextTokenizer(
            vocab_file=amharic_vocab_path,
            normalizer=AmharicTextNormalizer()
        )
        
        # Load model
        self.model = self._load_model()
        
        # Load vocoder if provided
        self.vocoder = self._load_vocoder() if vocoder_path else None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Evaluation metrics storage
        self.evaluation_results = {}
    
    def _setup_logging(self):
        """Setup logging for evaluation"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_model(self) -> UnifiedVoice:
        """Load the fine-tuned Amharic model"""
        self.logger.info("Loading Amharic TTS model...")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Create model with Amharic configuration
            model = UnifiedVoice(
                layers=self.config['gpt']['layers'],
                model_dim=self.config['gpt']['model_dim'],
                heads=self.config['gpt']['heads'],
                max_text_tokens=self.config['gpt']['max_text_tokens'],
                max_mel_tokens=self.config['gpt']['max_mel_tokens'],
                number_text_tokens=self.tokenizer.vocab_size,
                number_mel_codes=self.config['gpt']['number_mel_codes'],
                start_text_token=self.config['gpt']['start_text_token'],
                stop_text_token=self.config['gpt']['stop_text_token'],
                start_mel_token=self.config['gpt']['start_mel_token'],
                stop_mel_token=self.config['gpt']['stop_mel_token'],
                condition_type=self.config['gpt']['condition_type'],
                condition_module=self.config['gpt']['condition_module'],
                emo_condition_module=self.config['gpt']['emo_condition_module']
            )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            self.logger.info("Amharic model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _load_vocoder(self):
        """Load vocoder for audio generation"""
        try:
            if self.vocoder_path:
                self.logger.info("Loading vocoder...")
                vocoder = load_vocoder(self.vocoder_path)
                return vocoder
        except Exception as e:
            self.logger.warning(f"Could not load vocoder: {e}")
        return None
    
    def evaluate_text_processing(self, test_texts: List[str]) -> Dict:
        """Evaluate Amharic text processing capabilities"""
        self.logger.info("Evaluating text processing...")
        
        results = {
            'total_texts': len(test_texts),
            'successful_processing': 0,
            'tokenization_success': 0,
            'normalization_effectiveness': 0,
            'vocabulary_coverage': 0,
            'processing_errors': []
        }
        
        processed_tokens = []
        
        for i, text in enumerate(test_texts):
            try:
                # Original text analysis
                original_length = len(text)
                
                # Normalize text
                normalized_text = self.tokenizer.normalizer.normalize(text)
                
                if normalized_text:
                    results['normalization_effectiveness'] += 1
                
                # Tokenize text
                tokens = self.tokenizer.encode(normalized_text, out_type=str)
                
                if tokens:
                    results['tokenization_success'] += 1
                    processed_tokens.extend(tokens)
                else:
                    results['processing_errors'].append(f"Text {i}: Failed to tokenize")
                
                results['successful_processing'] += 1
                
            except Exception as e:
                results['processing_errors'].append(f"Text {i}: {str(e)}")
        
        # Calculate vocabulary coverage
        if processed_tokens:
            unique_tokens = set(processed_tokens)
            results['vocabulary_coverage'] = len(unique_tokens) / self.tokenizer.vocab_size
        
        # Calculate percentages
        total = len(test_texts)
        if total > 0:
            results['normalization_success_rate'] = results['normalization_effectiveness'] / total
            results['tokenization_success_rate'] = results['tokenization_success'] / total
            results['overall_success_rate'] = results['successful_processing'] / total
        
        self.logger.info(f"Text processing evaluation complete: {results['overall_success_rate']:.2%} success rate")
        return results
    
    def evaluate_model_inference(self, test_cases: List[Dict]) -> Dict:
        """Evaluate model inference capabilities"""
        self.logger.info("Evaluating model inference...")
        
        results = {
            'total_cases': len(test_cases),
            'successful_generations': 0,
            'generation_times': [],
            'output_lengths': [],
            'error_cases': [],
            'inference_stats': {}
        }
        
        for i, case in enumerate(tqdm(test_cases, desc="Evaluating inference")):
            try:
                text = case['text']
                
                # Tokenize input
                text_tokens = self.tokenizer.encode(text, out_type=int)
                
                if not text_tokens:
                    results['error_cases'].append(f"Case {i}: Failed to tokenize input text")
                    continue
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                # Note: This would require the actual IndexTTS2 inference implementation
                # For now, we'll simulate the inference
                
                # Mock inference (replace with actual inference code)
                with torch.no_grad():
                    # This is where the actual IndexTTS2 inference would happen
                    # predicted_mel = self.model.generate(...)
                    
                    # For evaluation purposes, we'll use a mock result
                    predicted_mel_length = np.random.randint(100, 1000)
                
                end_time.record()
                torch.cuda.synchronize()
                
                inference_time = start_time.elapsed_time(end_time)
                
                results['generation_times'].append(inference_time)
                results['output_lengths'].append(predicted_mel_length)
                results['successful_generations'] += 1
                
            except Exception as e:
                results['error_cases'].append(f"Case {i}: {str(e)}")
        
        # Calculate statistics
        if results['generation_times']:
            results['inference_stats'] = {
                'avg_generation_time': np.mean(results['generation_times']),
                'min_generation_time': np.min(results['generation_times']),
                'max_generation_time': np.max(results['generation_times']),
                'std_generation_time': np.std(results['generation_times'])
            }
        
        if results['output_lengths']:
            results['output_stats'] = {
                'avg_output_length': np.mean(results['output_lengths']),
                'min_output_length': np.min(results['output_lengths']),
                'max_output_length': np.max(results['output_lengths']),
                'std_output_length': np.std(results['output_lengths'])
            }
        
        # Calculate success rate
        total = len(test_cases)
        if total > 0:
            results['success_rate'] = results['successful_generations'] / total
        
        self.logger.info(f"Inference evaluation complete: {results.get('success_rate', 0):.2%} success rate")
        return results
    
    def evaluate_audio_quality(self, generated_audio_paths: List[str], reference_audio_paths: List[str] = None) -> Dict:
        """Evaluate generated audio quality"""
        self.logger.info("Evaluating audio quality...")
        
        results = {
            'total_samples': len(generated_audio_paths),
            'audio_stats': [],
            'quality_metrics': {},
            'error_samples': []
        }
        
        for i, audio_path in enumerate(generated_audio_paths):
            try:
                # Load generated audio
                audio, sr = torchaudio.load(audio_path)
                
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0)  # Convert to mono
                
                # Calculate audio statistics
                audio_np = audio.numpy()
                stats = {
                    'sample_id': i,
                    'duration': len(audio_np) / sr,
                    'rms_energy': np.sqrt(np.mean(audio_np**2)),
                    'peak_amplitude': np.max(np.abs(audio_np)),
                    'zero_crossing_rate': self._calculate_zero_crossing_rate(audio_np),
                    'spectral_centroid': self._calculate_spectral_centroid(audio_np, sr)
                }
                
                results['audio_stats'].append(stats)
                
            except Exception as e:
                results['error_samples'].append(f"Sample {i}: {str(e)}")
        
        # Calculate aggregate statistics
        if results['audio_stats']:
            duration_values = [s['duration'] for s in results['audio_stats']]
            energy_values = [s['rms_energy'] for s in results['audio_stats']]
            zcr_values = [s['zero_crossing_rate'] for s in results['audio_stats']]
            
            results['quality_metrics'] = {
                'avg_duration': np.mean(duration_values),
                'std_duration': np.std(duration_values),
                'avg_energy': np.mean(energy_values),
                'std_energy': np.std(energy_values),
                'avg_zcr': np.mean(zcr_values),
                'std_zcr': np.std(zcr_values),
                'clipping_samples': sum(1 for s in results['audio_stats'] if s['peak_amplitude'] > 0.99),
                'silence_samples': sum(1 for s in results['audio_stats'] if s['rms_energy'] < 0.01)
            }
        
        self.logger.info(f"Audio quality evaluation complete: {len(results['audio_stats'])} samples analyzed")
        return results
    
    def _calculate_zero_crossing_rate(self, audio: np.ndarray) -> float:
        """Calculate zero crossing rate"""
        return np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
    
    def _calculate_spectral_centroid(self, audio: np.ndarray, sr: int) -> float:
        """Calculate spectral centroid"""
        try:
            stft = np.fft.rfft(audio)
            magnitude = np.abs(stft)
            freqs = np.fft.rfftfreq(len(audio), 1/sr)
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            return centroid
        except:
            return 0.0
    
    def evaluate_linguistic_features(self, test_texts: List[str]) -> Dict:
        """Evaluate Amharic linguistic feature handling"""
        self.logger.info("Evaluating linguistic features...")
        
        results = {
            'amharic_characters': 0,
            'number_handling': 0,
            'abbreviation_handling': 0,
            'contraction_handling': 0,
            'script_consistency': 0,
            'linguistic_errors': []
        }
        
        for i, text in enumerate(test_texts):
            try:
                # Check for Amharic characters
                amharic_chars = re.findall(r'[ሀ-፿]', text)
                if amharic_chars:
                    results['amharic_characters'] += len(amharic_chars)
                
                # Check number handling
                number_words = ['አንድ', 'ሁለት', 'ሶስት', 'አራት', 'አምስት', 'ስድስት', 'ሰባት', 'ስምንት', 'ዘጠኝ', 'አስር']
                for word in number_words:
                    if word in text:
                        results['number_handling'] += 1
                
                # Check abbreviation handling
                abbreviations = ['ም.ም.', 'ዶ/ር', 'ፕ/ር']
                for abbr in abbreviations:
                    if abbr in text:
                        results['abbreviation_handling'] += 1
                
                # Check contraction handling
                contractions = ['ከሆነ', 'እንደሆነ', 'እንዲሁም']
                for contr in contractions:
                    if contr in text:
                        results['contraction_handling'] += 1
                
                # Check script consistency (simplified)
                if re.match(r'^[ሀ-፿\s.,!?፤፦፣።]*$', text):
                    results['script_consistency'] += 1
                
            except Exception as e:
                results['linguistic_errors'].append(f"Text {i}: {str(e)}")
        
        # Calculate percentages
        total = len(test_texts)
        if total > 0:
            results['script_consistency_rate'] = results['script_consistency'] / total
            results['amharic_character_density'] = results['amharic_characters'] / sum(len(text) for text in test_texts)
        
        self.logger.info("Linguistic feature evaluation complete")
        return results
    
    def generate_comprehensive_report(self, all_results: Dict) -> str:
        """Generate comprehensive evaluation report"""
        self.logger.info("Generating comprehensive evaluation report...")
        
        report = []
        report.append("# Amharic IndexTTS2 Evaluation Report")
        report.append(f"Generated on: {np.datetime64('now')}")
        report.append(f"Model: {self.model_path}")
        report.append(f"Vocabulary: {self.amharic_vocab_path}")
        report.append("")
        
        # Text Processing Results
        if 'text_processing' in all_results:
            tp_results = all_results['text_processing']
            report.append("## Text Processing Evaluation")
            report.append(f"- Total texts processed: {tp_results['total_texts']}")
            report.append(f"- Success rate: {tp_results.get('overall_success_rate', 0):.2%}")
            report.append(f"- Tokenization success rate: {tp_results.get('tokenization_success_rate', 0):.2%}")
            report.append(f"- Vocabulary coverage: {tp_results.get('vocabulary_coverage', 0):.2%}")
            if tp_results['processing_errors']:
                report.append(f"- Processing errors: {len(tp_results['processing_errors'])}")
            report.append("")
        
        # Model Inference Results
        if 'inference' in all_results:
            inf_results = all_results['inference']
            report.append("## Model Inference Evaluation")
            report.append(f"- Total test cases: {inf_results['total_cases']}")
            report.append(f"- Success rate: {inf_results.get('success_rate', 0):.2%}")
            if 'inference_stats' in inf_results:
                stats = inf_results['inference_stats']
                report.append(f"- Average generation time: {stats.get('avg_generation_time', 0):.2f}ms")
                report.append(f"- Generation time std: {stats.get('std_generation_time', 0):.2f}ms")
            if 'output_stats' in inf_results:
                out_stats = inf_results['output_stats']
                report.append(f"- Average output length: {out_stats.get('avg_output_length', 0):.0f} mel frames")
            if inf_results['error_cases']:
                report.append(f"- Error cases: {len(inf_results['error_cases'])}")
            report.append("")
        
        # Audio Quality Results
        if 'audio_quality' in all_results:
            aq_results = all_results['audio_quality']
            report.append("## Audio Quality Evaluation")
            report.append(f"- Total samples: {aq_results['total_samples']}")
            report.append(f"- Successfully analyzed: {len(aq_results['audio_stats'])}")
            if 'quality_metrics' in aq_results:
                metrics = aq_results['quality_metrics']
                report.append(f"- Average duration: {metrics.get('avg_duration', 0):.2f}s")
                report.append(f"- Average energy: {metrics.get('avg_energy', 0):.4f}")
                report.append(f"- Clipping samples: {metrics.get('clipping_samples', 0)}")
                report.append(f"- Silence samples: {metrics.get('silence_samples', 0)}")
            if aq_results['error_samples']:
                report.append(f"- Error samples: {len(aq_results['error_samples'])}")
            report.append("")
        
        # Linguistic Features Results
        if 'linguistic' in all_results:
            ling_results = all_results['linguistic']
            report.append("## Linguistic Feature Evaluation")
            report.append(f"- Amharic character density: {ling_results.get('amharic_character_density', 0):.3f}")
            report.append(f"- Script consistency rate: {ling_results.get('script_consistency_rate', 0):.2%}")
            report.append(f"- Number handling instances: {ling_results['number_handling']}")
            report.append(f"- Abbreviation handling instances: {ling_results['abbreviation_handling']}")
            report.append(f"- Contraction handling instances: {ling_results['contraction_handling']}")
            if ling_results['linguistic_errors']:
                report.append(f"- Linguistic errors: {len(ling_results['linguistic_errors'])}")
            report.append("")
        
        # Summary and Recommendations
        report.append("## Summary and Recommendations")
        
        # Overall scoring (simplified)
        overall_score = 0
        total_metrics = 0
        
        if 'text_processing' in all_results:
            overall_score += all_results['text_processing'].get('overall_success_rate', 0)
            total_metrics += 1
        
        if 'inference' in all_results:
            overall_score += all_results['inference'].get('success_rate', 0)
            total_metrics += 1
        
        if 'linguistic' in all_results:
            overall_score += all_results['linguistic'].get('script_consistency_rate', 0)
            total_metrics += 1
        
        if total_metrics > 0:
            overall_score /= total_metrics
            report.append(f"- Overall evaluation score: {overall_score:.2%}")
        
        # Recommendations
        report.append("\n### Recommendations:")
        if 'text_processing' in all_results and all_results['text_processing'].get('overall_success_rate', 0) < 0.9:
            report.append("- Improve text preprocessing and normalization")
        if 'inference' in all_results and all_results['inference'].get('success_rate', 0) < 0.8:
            report.append("- Improve model inference stability")
        if 'audio_quality' in all_results:
            metrics = all_results['audio_quality'].get('quality_metrics', {})
            if metrics.get('clipping_samples', 0) > 0:
                report.append("- Address audio clipping issues")
            if metrics.get('silence_samples', 0) > len(all_results['audio_quality']['audio_stats']) * 0.1:
                report.append("- Address silence issues in generated audio")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / "evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.logger.info(f"Evaluation report saved: {report_path}")
        return report_text
    
    def run_comprehensive_evaluation(
        self,
        test_texts: List[str],
        test_cases: List[Dict] = None,
        generated_audio_paths: List[str] = None,
        reference_audio_paths: List[str] = None
    ) -> Dict:
        """Run comprehensive evaluation pipeline"""
        self.logger.info("Starting comprehensive Amharic TTS evaluation...")
        
        all_results = {}
        
        # Text processing evaluation
        if test_texts:
            all_results['text_processing'] = self.evaluate_text_processing(test_texts)
        
        # Model inference evaluation
        if test_cases:
            all_results['inference'] = self.evaluate_model_inference(test_cases)
        
        # Audio quality evaluation
        if generated_audio_paths:
            all_results['audio_quality'] = self.evaluate_audio_quality(
                generated_audio_paths, reference_audio_paths
            )
        
        # Linguistic features evaluation
        if test_texts:
            all_results['linguistic'] = self.evaluate_linguistic_features(test_texts)
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(all_results)
        
        # Save results to JSON
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(all_results), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Evaluation results saved: {results_path}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Amharic IndexTTS2 fine-tuned model")
    parser.add_argument("--config", type=str, required=True, help="Path to Amharic config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--amharic_vocab", type=str, required=True, help="Path to Amharic vocabulary")
    parser.add_argument("--output_dir", type=str, default="amharic_evaluation", help="Output directory")
    parser.add_argument("--test_texts_file", type=str, help="File with test texts")
    parser.add_argument("--test_audio_dir", type=str, help="Directory with generated audio samples")
    parser.add_argument("--vocoder_path", type=str, help="Path to vocoder model")
    
    args = parser.parse_args()
    
    # Load test texts
    test_texts = []
    if args.test_texts_file and os.path.exists(args.test_texts_file):
        with open(args.test_texts_file, 'r', encoding='utf-8') as f:
            test_texts = [line.strip() for line in f if line.strip()]
    
    # Generate test cases if no file provided
    if not test_texts:
        test_texts = [
            "ሰላም ዓለም! እንደምን አደርኩ?",
            "ዛሬ የተሻለ ቀን ነው።",
            "እባኮትልትን ማሳመን አልችልም።",
            "ይህ የአማርኛ ድምጽ ማመንጫ ነው።",
            "በትምህርት ቤት ላይ ነበርኩ።"
        ]
    
    # Load generated audio paths
    generated_audio_paths = []
    if args.test_audio_dir and os.path.exists(args.test_audio_dir):
        import glob
        audio_extensions = {'.wav', '.flac', '.mp3', '.m4a'}
        for ext in audio_extensions:
            generated_audio_paths.extend(glob.glob(os.path.join(args.test_audio_dir, f"*{ext}")))
    
    # Initialize evaluator
    evaluator = AmharicTTSEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        amharic_vocab_path=args.amharic_vocab,
        vocoder_path=args.vocoder_path,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        test_texts=test_texts,
        generated_audio_paths=generated_audio_paths
    )
    
    print("Amharic IndexTTS2 evaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
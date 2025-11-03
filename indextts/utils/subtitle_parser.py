"""Professional subtitle parser for SRT/VTT formats with audio alignment"""
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    import pysrt
    PYSRT_AVAILABLE = True
except ImportError:
    PYSRT_AVAILABLE = False

try:
    import webvtt
    WEBVTT_AVAILABLE = True
except ImportError:
    WEBVTT_AVAILABLE = False


@dataclass
class SubtitleSegment:
    """Represents a single subtitle segment with timing"""
    index: int
    start_time: float  # in seconds
    end_time: float    # in seconds
    text: str
    duration: float
    
    def __post_init__(self):
        self.duration = self.end_time - self.start_time


class SubtitleParser:
    """Parse SRT/VTT subtitle files and align with audio"""
    
    def __init__(self, preserve_punctuation: bool = True):
        self.preserve_punctuation = preserve_punctuation
        self.logger = logging.getLogger(__name__)
    
    def parse_srt(self, srt_path: str) -> List[SubtitleSegment]:
        """Parse SRT file and extract segments with timestamps"""
        if not PYSRT_AVAILABLE:
            return self._parse_srt_manual(srt_path)
        
        try:
            subs = pysrt.open(srt_path, encoding='utf-8')
            segments = []
            
            for i, sub in enumerate(subs):
                start_seconds = self._time_to_seconds(sub.start)
                end_seconds = self._time_to_seconds(sub.end)
                text = sub.text.replace('\n', ' ').strip()
                
                if text:  # Skip empty subtitles
                    segments.append(SubtitleSegment(
                        index=i,
                        start_time=start_seconds,
                        end_time=end_seconds,
                        text=text,
                        duration=end_seconds - start_seconds
                    ))
            
            self.logger.info(f"Parsed {len(segments)} segments from SRT file")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error parsing SRT with pysrt: {e}")
            return self._parse_srt_manual(srt_path)
    
    def parse_vtt(self, vtt_path: str) -> List[SubtitleSegment]:
        """Parse VTT (WebVTT) file and extract segments"""
        if not WEBVTT_AVAILABLE:
            return self._parse_vtt_manual(vtt_path)
        
        try:
            segments = []
            for i, caption in enumerate(webvtt.read(vtt_path)):
                start_seconds = self._vtt_time_to_seconds(caption.start)
                end_seconds = self._vtt_time_to_seconds(caption.end)
                text = caption.text.replace('\n', ' ').strip()
                
                if text:
                    segments.append(SubtitleSegment(
                        index=i,
                        start_time=start_seconds,
                        end_time=end_seconds,
                        text=text,
                        duration=end_seconds - start_seconds
                    ))
            
            self.logger.info(f"Parsed {len(segments)} segments from VTT file")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error parsing VTT with webvtt: {e}")
            return self._parse_vtt_manual(vtt_path)
    
    def _parse_srt_manual(self, srt_path: str) -> List[SubtitleSegment]:
        """Manual SRT parsing fallback"""
        segments = []
        
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # SRT format: index, timestamp, text, blank line
        pattern = r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})\s+([\s\S]*?)(?=\n\n|\Z)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            index, start, end, text = match
            start_sec = self._parse_srt_time(start)
            end_sec = self._parse_srt_time(end)
            clean_text = text.replace('\n', ' ').strip()
            
            if clean_text:
                segments.append(SubtitleSegment(
                    index=int(index) - 1,
                    start_time=start_sec,
                    end_time=end_sec,
                    text=clean_text,
                    duration=end_sec - start_sec
                ))
        
        return segments
    
    def _parse_vtt_manual(self, vtt_path: str) -> List[SubtitleSegment]:
        """Manual VTT parsing fallback"""
        segments = []
        
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        index = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for timestamp line
            if '-->' in line:
                times = line.split('-->')
                if len(times) == 2:
                    start_sec = self._parse_vtt_time(times[0].strip())
                    end_sec = self._parse_vtt_time(times[1].strip().split()[0])  # Remove optional settings
                    
                    # Collect text lines
                    i += 1
                    text_lines = []
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    text = ' '.join(text_lines)
                    if text:
                        segments.append(SubtitleSegment(
                            index=index,
                            start_time=start_sec,
                            end_time=end_sec,
                            text=text,
                            duration=end_sec - start_sec
                        ))
                        index += 1
            
            i += 1
        
        return segments
    
    def _time_to_seconds(self, time_obj) -> float:
        """Convert pysrt time object to seconds"""
        return time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds + time_obj.milliseconds / 1000.0
    
    def _vtt_time_to_seconds(self, time_str: str) -> float:
        """Convert VTT time string to seconds"""
        return self._parse_vtt_time(time_str)
    
    def _parse_srt_time(self, time_str: str) -> float:
        """Parse SRT time format: 00:00:00,000"""
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def _parse_vtt_time(self, time_str: str) -> float:
        """Parse VTT time format: 00:00:00.000 or 00:00.000"""
        parts = time_str.strip().split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return 0.0
    
    def extract_audio_segments(
        self,
        audio_path: str,
        segments: List[SubtitleSegment],
        output_dir: str,
        format: str = 'wav'
    ) -> List[Dict]:
        """Extract audio segments using FFmpeg based on subtitle timestamps"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extracted = []
        
        for seg in segments:
            output_file = output_path / f"segment_{seg.index:05d}.{format}"
            
            # FFmpeg command for precise audio extraction
            cmd = [
                'ffmpeg',
                '-i', audio_path,
                '-ss', str(seg.start_time),
                '-t', str(seg.duration),
                '-acodec', 'pcm_s16le' if format == 'wav' else 'copy',
                '-ar', '24000',  # 24kHz sample rate
                '-ac', '1',       # Mono
                '-y',             # Overwrite
                str(output_file)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                extracted.append({
                    'audio_path': str(output_file),
                    'text': seg.text,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'duration': seg.duration,
                    'segment_index': seg.index
                })
            except subprocess.CalledProcessError as e:
                self.logger.error(f"FFmpeg extraction failed for segment {seg.index}: {e}")
                continue
        
        self.logger.info(f"Successfully extracted {len(extracted)}/{len(segments)} segments")
        return extracted
    
    def validate_timing(self, segments: List[SubtitleSegment]) -> Tuple[bool, List[str]]:
        """Validate subtitle timing consistency"""
        issues = []
        
        for i, seg in enumerate(segments):
            # Check for negative duration
            if seg.duration <= 0:
                issues.append(f"Segment {i}: Invalid duration ({seg.duration}s)")
            
            # Check for very short segments (<0.1s)
            if seg.duration < 0.1:
                issues.append(f"Segment {i}: Very short duration ({seg.duration}s)")
            
            # Check for overlap with next segment
            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                if seg.end_time > next_seg.start_time:
                    overlap = seg.end_time - next_seg.start_time
                    issues.append(f"Segment {i}: Overlaps with next by {overlap:.3f}s")
            
            # Check for very long segments (>30s)
            if seg.duration > 30:
                issues.append(f"Segment {i}: Very long duration ({seg.duration}s)")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def parse_file(self, subtitle_path: str) -> List[SubtitleSegment]:
        """Auto-detect and parse SRT or VTT file"""
        path = Path(subtitle_path)
        
        if path.suffix.lower() == '.srt':
            return self.parse_srt(subtitle_path)
        elif path.suffix.lower() == '.vtt':
            return self.parse_vtt(subtitle_path)
        else:
            # Try to auto-detect
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            if first_line.upper() == 'WEBVTT':
                return self.parse_vtt(subtitle_path)
            else:
                return self.parse_srt(subtitle_path)

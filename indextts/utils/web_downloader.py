"""Professional web media downloader with yt-dlp integration"""
import subprocess
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class DownloadResult:
    """Result of a download operation"""
    url: str
    success: bool
    audio_path: Optional[str] = None
    subtitle_path: Optional[str] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None


class WebMediaDownloader:
    """Download media and subtitles from web URLs using yt-dlp"""
    
    def __init__(
        self,
        output_dir: str = "downloads",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 300
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Check if yt-dlp is available
        self._check_ytdlp()
    
    def _check_ytdlp(self) -> bool:
        """Check if yt-dlp is installed"""
        try:
            result = subprocess.run(
                ['yt-dlp', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.logger.info(f"yt-dlp version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.error("yt-dlp not found. Install with: pip install yt-dlp")
            return False
    
    def download_from_url(
        self,
        url: str,
        extract_audio: bool = True,
        extract_subtitles: bool = True,
        audio_format: str = 'wav',
        sample_rate: int = 24000,
        custom_filename: Optional[str] = None
    ) -> DownloadResult:
        """Download media from a single URL with retries"""
        for attempt in range(self.max_retries):
            try:
                result = self._download_single(
                    url, extract_audio, extract_subtitles,
                    audio_format, sample_rate, custom_filename
                )
                
                if result.success:
                    return result
                
                # Wait before retry
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {delay}s")
                    time.sleep(delay)
                    
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return DownloadResult(url=url, success=False, error=str(e))
        
        return DownloadResult(url=url, success=False, error="Max retries exceeded")
    
    def _download_single(
        self,
        url: str,
        extract_audio: bool,
        extract_subtitles: bool,
        audio_format: str,
        sample_rate: int,
        custom_filename: Optional[str]
    ) -> DownloadResult:
        """Download from URL without retry logic"""
        # Generate output filename
        if custom_filename:
            output_template = str(self.output_dir / custom_filename)
        else:
            output_template = str(self.output_dir / "%(id)s.%(ext)s")
        
        # Build yt-dlp command
        cmd = ['yt-dlp']
        
        if extract_audio:
            cmd.extend([
                '-x',  # Extract audio
                '--audio-format', audio_format,
                '--audio-quality', '0',  # Best quality
                '--postprocessor-args', f"ffmpeg:-ar {sample_rate} -ac 1",  # Mono, target sample rate
            ])
        
        if extract_subtitles:
            cmd.extend([
                '--write-auto-sub',  # Auto-generated subtitles
                '--write-sub',       # Manual subtitles
                '--sub-lang', 'en,am,de',  # Multiple languages
                '--sub-format', 'srt/vtt/best',
                '--convert-subs', 'srt',  # Convert to SRT
            ])
        
        cmd.extend([
            '--write-info-json',  # Save metadata
            '-o', output_template,
            '--no-playlist',      # Don't download playlists
            '--no-warnings',
            url
        ])
        
        # Execute download
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                return DownloadResult(
                    url=url,
                    success=False,
                    error=result.stderr
                )
            
            # Find downloaded files
            audio_path = self._find_downloaded_file(audio_format)
            subtitle_path = self._find_downloaded_file('srt')
            metadata = self._load_metadata()
            
            return DownloadResult(
                url=url,
                success=True,
                audio_path=audio_path,
                subtitle_path=subtitle_path,
                metadata=metadata
            )
            
        except subprocess.TimeoutExpired:
            return DownloadResult(
                url=url,
                success=False,
                error=f"Download timeout after {self.timeout}s"
            )
        except Exception as e:
            return DownloadResult(
                url=url,
                success=False,
                error=str(e)
            )
    
    def _find_downloaded_file(self, extension: str) -> Optional[str]:
        """Find most recently downloaded file with given extension"""
        files = list(self.output_dir.glob(f"*.{extension}"))
        if not files:
            return None
        # Return most recent file
        latest = max(files, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def _load_metadata(self) -> Optional[Dict]:
        """Load JSON metadata if available"""
        json_files = list(self.output_dir.glob("*.info.json"))
        if not json_files:
            return None
        
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None
    
    def batch_download(
        self,
        urls: List[str],
        max_workers: int = 4,
        extract_audio: bool = True,
        extract_subtitles: bool = True,
        audio_format: str = 'wav',
        sample_rate: int = 24000
    ) -> List[DownloadResult]:
        """Download multiple URLs in parallel"""
        self.logger.info(f"Starting batch download of {len(urls)} URLs with {max_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all downloads
            future_to_url = {
                executor.submit(
                    self.download_from_url,
                    url, extract_audio, extract_subtitles,
                    audio_format, sample_rate
                ): url
                for url in urls
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        self.logger.info(f"✅ Downloaded: {url}")
                    else:
                        self.logger.error(f"❌ Failed: {url} - {result.error}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Exception for {url}: {e}")
                    results.append(DownloadResult(url=url, success=False, error=str(e)))
        
        success_count = sum(1 for r in results if r.success)
        self.logger.info(f"Batch download complete: {success_count}/{len(urls)} successful")
        
        return results
    
    def download_from_file(
        self,
        url_file: str,
        max_workers: int = 4,
        **kwargs
    ) -> List[DownloadResult]:
        """Download URLs from a text file (one URL per line)"""
        with open(url_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        self.logger.info(f"Loaded {len(urls)} URLs from {url_file}")
        return self.batch_download(urls, max_workers=max_workers, **kwargs)
    
    def get_video_info(self, url: str) -> Optional[Dict]:
        """Get video information without downloading"""
        cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-playlist',
            url
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                self.logger.error(f"Failed to get info: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting video info: {e}")
            return None

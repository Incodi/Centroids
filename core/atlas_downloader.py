from datetime import datetime, timedelta
import time
import select
import json
import os
import queue
import re
import socket
import subprocess
import threading
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
try:
    from .vtt_helper import VTTSegment
except ImportError:
    from vtt_helper import VTTSegment


class SubtitleDownloader(tk.Toplevel):
    
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.title("Subtitle Downloader")
        self.minsize(820, 600)
        self.center_window(1000, 750)
        
        self.max_downloads = 300
        self.current_downloads = 0
        self.batch_filter_handles = []
        self.batch_download_number = 150
        
        # Threading
        self.worker_thread = None
        self.should_stop = threading.Event()
        self.messages = queue.Queue()
        self._metadata_cache = {}
        self._metadata_id_cache = {}

        # Style
        self.style = ttk.Style(self)
        #self.style.theme_use('clam')
        
        # Widget setup
        self.create_widgets()
        
        # Start checking for messages from worker thread
        self.after(100, self.check_messages)
    
    def center_window(self, width, height):
        """Size the window and place it near the centre of the screen"""
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 3, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Create all UI elements"""
        # Main container
        container = ttk.Frame(self, padding="20")
        container.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Let the output area take up the extra vertical space
        container.columnconfigure(0, weight=1)
        container.rowconfigure(8, weight=1)
        
        # URL input section
        self.create_url_input(container)
        
        # Options checkboxes
        self.create_options(container)
        
        # Action buttons
        self.create_buttons(container)
        
        # Progress bar 
        self.style.configure("red.Horizontal.TProgressbar", background='red', foreground='black')
        
        self.progress_bar = ttk.Progressbar(container, style="red.Horizontal.TProgressbar", mode='determinate')
        self.progress_bar.grid(column=0, row=6, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)) # under action buttons
        
        # Status label
        self.status_text = tk.StringVar()
        ttk.Label(container, textvariable=self.status_text).grid(
            column=0, row=7, columnspan=3, sticky=tk.W, pady=(10, 0)) # under progress bar
        
        # Output text area with scrollbar
        self.create_output_area(container)
    
    def create_url_input(self, parent):
        """Create URL input field"""
        ttk.Label(parent, text="YouTube Channel URL or Playlist URL:").grid(
            column=0, row=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        self.url_input = ttk.Entry(parent, width=100) # Entry is the place where users can input their URL
        self.url_input.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)) # row 1 means it is udner row 0 (label)
    
    def create_options(self, parent):
        """Create option checkboxes: Metadata update and batch download (flawed due to yt-dlp limitations)"""
        # Metadata only checkbox
        self.metadata_only = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Update metadata only (doesn't download new subtitles)",
            variable=self.metadata_only
        ).grid(column=0, row=2, columnspan=3, sticky=tk.W, pady=(5, 10)) # is under URL entry
        
        # Batch options row: batch mode, Z-0 reverse order and handle list button
        batch_frame = ttk.Frame(parent)
        batch_frame.grid(column=0, row=3, columnspan=3, sticky=tk.W, pady=(5, 10)) # is under metadata checkbox
        
        # Batch download checkbox
        self.batch_mode = tk.BooleanVar()
        ttk.Checkbutton(
            batch_frame,
            text=f"Batch download: {self.batch_download_number} videos from each channel in input directory", # BATCH DOWNLOAD NUMBER batch 5
            variable=self.batch_mode
        ).grid(column=0, row=0, sticky=tk.W)
        
        # Z-0 (reverse order) checkbox
        self.reverse_batch_order = tk.BooleanVar()
        ttk.Checkbutton(
            batch_frame,
            text="Z\u20130 (reverse order)",
            variable=self.reverse_batch_order
        ).grid(column=1, row=0, sticky=tk.W, padx=(16, 0))
        
        self.batch_filter_button = ttk.Button(
            batch_frame,
            text="Load Handle List (.txt)",
            command=self.load_batch_handle_list
        )
        self.batch_filter_button.grid(column=2, row=0, sticky=tk.W, padx=(16, 0))
    
    def create_buttons(self, parent):
        """Create action buttons: Download/Update metadata and Check Consistency"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(column=0, row=4, columnspan=3, sticky=tk.W, pady=(10, 0)) # is under options checkboxes
        
        self.download_button = ttk.Button(
            button_frame,
            text="Download Subtitles",
            command=self.start_download
        )
        self.download_button.grid(column=0, row=0, sticky=tk.W) 
        # column 0 means it is on the left side
        
        self.consistency_button = ttk.Button(
            button_frame,
            text="Check Consistency",
            command=self.start_consistency_check
        )
        self.consistency_button.grid(column=1, row=0, sticky=tk.W, padx=(20, 0)) 
        # column 1 means it is on the right side by download button
    
    def create_output_area(self, parent):
        """Create output text area with scrollbar"""
        output_frame = ttk.Frame(parent)
        output_frame.grid(column=0, row=8, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.output = tk.Text(output_frame, height=18, wrap=tk.WORD) # is text area for output logs
        self.output.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output.yview) # you can scroll here!
        scrollbar.grid(column=1, row=0, sticky=(tk.N, tk.S)) # scrollbar is on the right side of output text area
        self.output['yscrollcommand'] = scrollbar.set
    
    # Main Control Functions ⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚
    
    def start_download(self):
        """ Start the download process in a background thread"""
        if self.is_working():
            messagebox.showwarning("Warning", "Operation already in progress!", parent=self)
            return
        
        self.should_stop.clear()
        self.prepare_for_download()
        self.worker_thread = threading.Thread(target=self.download_worker, daemon=True)
        # uses a thread to run download_worker function in the background
        self.worker_thread.start()
    
    def start_consistency_check(self):
        """Start the consistency check in a background thread"""
        if self.is_working():
            messagebox.showwarning("Warning", "Operation already in progress!", parent=self)
            return
        
        self.should_stop.clear()
        self.prepare_for_consistency_check()
        self.worker_thread = threading.Thread(target=self.check_consistency_worker, daemon=True)
        self.worker_thread.start()
    
    def is_working(self): # fallback in case of rare situation
        """Check if a background task is running"""
        return self.worker_thread and self.worker_thread.is_alive()
    
    def prepare_for_download(self):
        """Reset UI for download operation"""
        self.disable_buttons()
        self.progress_bar['value'] = 0
        self.output.delete(1.0, tk.END)
        self.current_downloads = 0 # reset current downloads count!
        
        if self.batch_mode.get():
            button_text = "Batch Downloading..."
        elif self.metadata_only.get():
            button_text = "Updating Metadata..."
        else:
            button_text = "Downloading..."
        
        self.download_button.config(text=button_text)
    
    def prepare_for_consistency_check(self):
        """Reset UI for consistency check"""
        self.disable_buttons()
        self.consistency_button.config(text="Checking...")
        self.progress_bar['value'] = 0
        self.output.delete(1.0, tk.END)
        self.status_text.set("Starting consistency check...")
    
    def disable_buttons(self):
        """Disable action buttons during operation"""
        self.download_button.config(state=tk.DISABLED)
        self.consistency_button.config(state=tk.DISABLED)
        self.batch_filter_button.config(state=tk.DISABLED)
    
    def enable_buttons(self):
        """Re-enable action buttons after operation"""
        self.download_button.config(state=tk.NORMAL, text="Download Subtitles")
        self.consistency_button.config(state=tk.NORMAL, text="Check Consistency")
        self.batch_filter_button.config(state=tk.NORMAL)
    
    # Worker Thread Functions ⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚
    
    def download_worker(self):
        """Background worker for download operations"""
        try:
            # Check if yt-dlp is available
            if not self.is_yt_dlp_available():
                self.send_message('error', "yt-dlp is not installed or not found in PATH. Please install yt-dlp first.")
                return

            if self.batch_mode.get():
                self.process_batch_download()
            else:
                url = self.url_input.get().strip()
                if not url:
                    self.send_message('error', "Please enter a YouTube channel or playlist URL!")
                    return

                channel_id = self.extract_channel_id(url)
                folder = self.get_channel_folder(channel_id)

                if self.metadata_only.get():
                    self.update_metadata(folder)
                else:
                    self.download_from_channel(folder, url)

        except Exception as e:
            self.send_message('error', f"Unexpected error: {str(e)}")
        finally:
            self.send_message('done', None)
    
    def check_consistency_worker(self):
        """Background worker for consistency check"""
        try:
            input_folder = self.get_input_folder()
            if not os.path.exists(input_folder):
                self.send_message('error', f"Input directory not found: {input_folder}")
                return
            
            channels = self.get_all_channels(input_folder)
            if not channels:
                self.send_message('error', "No channel directories found")
                return
            
            self.send_message('status', f"Checking {len(channels)} channels...")
            
            total_issues = 0
            total_fixed = 0
            
            for i, channel in enumerate(channels):
                if self.should_stop.is_set():
                    break
                
                folder = os.path.join(input_folder, channel)
                issues, fixed = self.check_channel(folder, channel)
                total_issues += issues
                total_fixed += fixed
                
                progress = ((i + 1) / len(channels)) * 100
                self.send_message('progress', progress)
            
            self.send_message('status', 
                f"✓ Found {total_issues} issues, fixed {total_fixed}")
        
        except Exception as e:
            self.send_message('error', f"Consistency check error: {str(e)}")
        finally:
            self.send_message('done', None)
    
    # Batch Download Functions ⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚
    
    def process_batch_download(self):
        """Download from all channels in input directory"""
        # Check if yt-dlp is available
        if not self.is_yt_dlp_available():
            self.send_message('error', "yt-dlp is not installed or not found in PATH. Please install yt-dlp first.")
            return

        input_folder = self.get_input_folder()
        if not os.path.exists(input_folder):
            self.send_message('error', f"Input directory not found: {input_folder}")
            return

        reverse_order = self.reverse_batch_order.get()

        channels = self.get_all_channels(input_folder)
        channels.sort(key=str.lower)
        if not channels:
            self.send_message('error', "No channel directories found")
            return

        channels = self.apply_batch_filter(channels)
        if not channels:
            self.send_message('error', "No channels matched the loaded handle list")
            return

        # Z-0 option: process the final list back-to-front. Applied after the
        # handle filter so the checkbox works with or without a loaded list.
        if reverse_order:
            channels.reverse()

        order_label = "Z→0" if reverse_order else "0→Z"
        self.send_message('status', f"Processing {len(channels)} channels ({order_label})")

        for i, channel in enumerate(channels):
            if self.should_stop.is_set():
                break

            folder = os.path.join(input_folder, channel)
            self.process_single_channel_batch(folder, channel, i + 1, len(channels))

            progress = ((i + 1) / len(channels)) * 100 # progress bar
            self.send_message('progress', progress)

        self.send_message('status', f"✓ Batch complete. Processed {len(channels)} channels")

    def load_batch_handle_list(self):
        """Load a text file with one handle per line for filtered batch downloads"""
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Select channel handle list",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {str(e)}", parent=self)
            return

        handles = self.parse_handles_from_text(text)
        if not handles:
            messagebox.showwarning(
                "No Handles Found",
                "No valid channel handles were found in the selected file.",
                parent=self
            )
            return

        # Normalize and handle creation for new channels provided as
        # `youtube.com/@Handle` (with or without protocol/ www).
        created = 0
        normalized_handles = []
        pattern = re.compile(r'^(?:https?://)?(?:www\.)?youtube\.com/@(?P<handle>[A-Za-z0-9_\-\.]+)/*$', re.IGNORECASE)

        for h in handles:
            m = pattern.match(h.strip())
            if m:
                handle_name = m.group('handle')
                # Ensure input folder exists for this handle and create minimal metadata
                folder = os.path.join(self.get_input_folder(), handle_name)
                try:
                    if not os.path.exists(folder):
                        os.makedirs(folder, exist_ok=True)
                        created += 1

                    metadata_file = os.path.join(folder, 'metadata.json')
                    if not os.path.exists(metadata_file):
                        # Save as a list to match expected metadata format
                        try:
                            self.save_json(metadata_file, [self._make_minimal_channel_metadata(handle_name)])
                        except Exception as e:
                            self.send_message('warning', f"Could not write metadata for {handle_name}: {str(e)}")
                    else:
                        try:
                            existing_metadata = self.load_json(metadata_file)
                        except Exception:
                            existing_metadata = None

                        if not existing_metadata:
                            try:
                                self.save_json(metadata_file, [self._make_minimal_channel_metadata(handle_name)])
                            except Exception as e:
                                self.send_message('warning', f"Could not seed metadata for {handle_name}: {str(e)}")

                    # Use the directory name for filtering (so it matches get_all_channels)
                    normalized_handles.append(handle_name)
                    continue
                except Exception as e:
                    # Log but continue processing other handles
                    self.send_message('log', f"Error preparing folder for {h}: {str(e)}")

            # Not a youtube.com/@... URL - keep as provided
            normalized_handles.append(h)

        self.batch_filter_handles = normalized_handles
        self.batch_filter_button.config(text=f"Handle List: {len(normalized_handles)}")
        if created:
            self.status_text.set(f"Loaded {len(normalized_handles)} handles ({created} new channels created)")
        else:
            self.status_text.set(f"Loaded {len(normalized_handles)} handles for batch filter")

    def parse_handles_from_text(self, text):
        """Parse channel handles from plain lines or '[x/y] Processing: HANDLE' lines"""
        handles = []
        seen = set()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if 'processing:' in line.lower():
                line = re.split(r'processing:\s*', line, maxsplit=1, flags=re.IGNORECASE)[-1].strip()

            handle = line.strip().rstrip(',;')
            if not handle:
                continue

            key = handle.casefold()
            if key in seen:
                continue

            seen.add(key)
            handles.append(handle)

        return handles

    def apply_batch_filter(self, channels):
        """Apply optional handle filter to available channel directories"""
        if not self.batch_filter_handles:
            return channels

        selected = []
        available_map = {name.casefold(): name for name in channels}

        for handle in self.batch_filter_handles:
            match = available_map.get(handle.casefold())
            if match:
                selected.append(match)

        missing = [h for h in self.batch_filter_handles if h.casefold() not in available_map]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "..." if len(missing) > 8 else ""
            self.send_message('log', f"Handle list entries not found in input directory: {preview}{suffix}")

        self.send_message('log', f"Batch filter active: selected {len(selected)} of {len(channels)} channels")
        return selected
    
    def build_channel_url(self, channel_name, metadata):
        """Build a YouTube URL from channel metadata"""
        if not metadata:
            return None
        
        # Try to use the first valid channel URL from metadata
        for entry in metadata:
            if isinstance(entry, dict):
                # Prefer direct channel_url
                channel_url = entry.get('channel_url', '').strip()
                if channel_url:
                    return channel_url
        
        # Fallback: try to build URL from channel_id
        for entry in metadata:
            if isinstance(entry, dict):
                channel_id = entry.get('channel_id', '').strip()
                if channel_id:
                    return f"https://www.youtube.com/channel/{channel_id}"
        
        # Last fallback: use channel name to construct URL
        if channel_name:
            # Assume channel_name might be a custom handle or username
            return f"https://www.youtube.com/c/{channel_name}"
        
        return None

    def _make_minimal_channel_metadata(self, channel_name):
        """Create a minimal metadata entry for a channel folder or handle."""
        safe_name = channel_name.strip().lstrip('@')
        is_channel_id = safe_name.startswith('UC') and len(safe_name) >= 24

        if is_channel_id:
            channel_url = f"https://www.youtube.com/channel/{safe_name}"
            channel_id = safe_name
        else:
            channel_url = f"https://www.youtube.com/@{safe_name}"
            channel_id = ''

        return {
            'id': safe_name,
            'title': safe_name,
            'url': channel_url,
            'timestamp': datetime.now().isoformat(),
            'channel_name': safe_name,
            'channel_url': channel_url,
            'channel_id': channel_id,
        }
    
    def process_single_channel_batch(self, folder, channel_name, current, total):
        """Process one channel in batch mode"""
        metadata_file = os.path.join(folder, "metadata.json")
        try:
            if not os.path.exists(metadata_file):
                self.send_message('warning', f"No metadata found for {channel_name}; seeding minimal metadata")
                metadata = [self._make_minimal_channel_metadata(channel_name)]
                try:
                    self.save_json(metadata_file, metadata)
                except Exception as e:
                    self.send_message('warning', f"Could not write fallback metadata for {channel_name}: {str(e)}")
            else:
                try:
                    metadata = self.load_json(metadata_file)
                except Exception as e:
                    self.send_message('warning', f"Could not read metadata for {channel_name}; using fallback URL: {str(e)}")
                    metadata = []

                if not metadata:
                    self.send_message('warning', f"Empty metadata for {channel_name}; seeding minimal metadata")
                    metadata = [self._make_minimal_channel_metadata(channel_name)]
                    try:
                        self.save_json(metadata_file, metadata)
                    except Exception as e:
                        self.send_message('warning', f"Could not update empty metadata for {channel_name}: {str(e)}")

            url = self.build_channel_url(channel_name, metadata)
            if not url:
                self.send_message('log', f"Skipping {channel_name}: Could not build URL")
                return

            self.send_message('log', f"[{current}/{total}] Processing: {channel_name}")

            # Use smaller limit for batch mode
            old_limit = self.max_downloads
            self.max_downloads = self.batch_download_number # BATCH DOWNLOAD NUMBER usually batch 5
            try:
                self.download_from_channel(folder, url)
            finally:
                self.max_downloads = old_limit

        except Exception as e:
            self.send_message('log', f"Error processing {channel_name}: {str(e)}")
    
    def download_from_channel(self, folder, url):
        """Download subtitles from a YouTube channel or playlist"""
        self._prepare_download_state(folder)

        # Create vtt_files subfolder for staging
        vtt_folder = os.path.join(folder, "vtt_files")
        os.makedirs(vtt_folder, exist_ok=True)
        
        # Build yt-dlp command
        command = self.build_download_command(vtt_folder, url)
        
        if not self.batch_mode.get():
            self.send_message('status', f"Starting download (max {self.max_downloads} subtitles)...")
        
        try:
            # Run yt-dlp and process output
            self.run_download_process(command, folder)
        finally:
            # Persist metadata once per channel instead of once per video.
            self._flush_metadata_cache(folder)

            # Finalize: convert VTT to VTX and clean up
            self._finalize_vtx_transcripts(folder)

    def _prepare_download_state(self, folder):
        """Load channel state and make metadata/archive agree before downloading."""
        _, metadata = self.load_metadata_ids(folder)
        metadata_ids = {
            entry['id'] for entry in metadata
            if self._is_video_metadata_entry(entry)
        }
        archive_ids, _ = self.load_archive_ids(folder)

        # metadata.json is authoritative for videos already known to Centroids.
        # Adding those IDs to the yt-dlp archive prevents duplicate downloads when
        # archive.txt was copied, truncated, or generated by another downloader.
        missing_archive = metadata_ids - archive_ids
        if missing_archive:
            self.add_archive_entries(folder, sorted(missing_archive))

        cache_key = os.path.abspath(folder)
        self._metadata_cache[cache_key] = list(metadata)
        self._metadata_id_cache[cache_key] = {
            entry.get('id') for entry in metadata
            if isinstance(entry, dict) and entry.get('id')
        }

    @staticmethod
    def _is_video_metadata_entry(entry):
        """Distinguish video records from the minimal channel seed record."""
        if not isinstance(entry, dict) or not entry.get('id'):
            return False
        if entry.get('upload_date') or entry.get('duration') is not None:
            return True
        return '/watch?' in entry.get('url', '')
    
    def _finalize_vtx_transcripts(self, folder):
        """
        Convert downloaded VTT files to VTX format and clean up.
        Only deletes VTT files that have been successfully converted to VTX.
        Leaves unconfirmed VTTs in the stage directory with a warning.
        """
        vtt_dir = os.path.join(folder, "vtt_files")
        vtx_dir = os.path.join(folder, "vtx_files")
        txt_dir = os.path.join(folder, "txt_files")
        
        if not os.path.exists(vtt_dir):
            return
        
        # Create vtx_dir if it doesn't exist
        os.makedirs(vtx_dir, exist_ok=True)
        
        # Convert VTT to VTX
        converter = VTTSegment()
        result = converter.convert_vtt_directory_to_vtx(vtt_dir, vtx_dir)
        
        # Report results
        if result['converted']:
            self.send_message('log', f"  Converted {len(result['converted'])} VTT files to VTX")
        
        if result['failed']:
            self.send_message('log', f"  Warning: {len(result['failed'])} VTT files failed to convert")
        
        # Verified deletion: only delete VTT if VTX exists and TXT exists and is non-empty
        deleted_count = 0
        for vtt_file in result['converted']:
            base_name = os.path.splitext(vtt_file)[0]
            vtx_file = os.path.join(vtx_dir, f"{base_name}.vtx")
            txt_file = os.path.join(txt_dir, f"{base_name}.txt")
            vtt_path = os.path.join(vtt_dir, vtt_file)
            
            # Only delete if VTX exists and TXT exists and is non-empty
            if os.path.exists(vtx_file) and os.path.exists(txt_file):
                try:
                    if os.path.getsize(txt_file) > 0:
                        os.remove(vtt_path)
                        deleted_count += 1
                except Exception as e:
                    self.send_message('log', f"  Warning: Could not delete {vtt_file}: {str(e)}")
        
        if deleted_count > 0:
            self.send_message('log', f"  Deleted {deleted_count} verified VTT files")
        
        # Check for unconfirmed VTTs (VTTs without VTX or without TXT)
        remaining_vtt = [f for f in os.listdir(vtt_dir) if f.endswith('.vtt')] if os.path.exists(vtt_dir) else []
        if remaining_vtt:
            self.send_message('log', f"  Warning: {len(remaining_vtt)} unconfirmed VTT files remain in vtt_files/")
            self.send_message('log', f"  These will be kept for manual review or retry")
        
        # Remove empty vtt_files directory
        if os.path.exists(vtt_dir) and not os.listdir(vtt_dir):
            try:
                os.rmdir(vtt_dir)
                self.send_message('log', f"  Removed empty vtt_files directory")
            except Exception:
                pass
    
    def build_download_command(self, vtt_folder, url):
        """Build the yt-dlp command with all options"""
        archive_file = os.path.join(os.path.dirname(vtt_folder), "archive.txt")
        output_template = os.path.join(vtt_folder, "%(title)s [%(id)s].%(ext)s")
        
        command = [
            self.get_yt_dlp_path(),
            "--write-auto-sub",
            "--sub-lang", "en",
            "--skip-download",
            "--convert-subs", "vtt",
            "--print-json",
            "--download-archive", archive_file,
            "--no-warnings",
            "--force-write-archive",
            "--no-overwrites",
            #"--sleep-subtitles", "1",
            "--extractor-args", "youtube:player-client=default,mweb",
            "-o", output_template,
            #"--cookies-from-browser", "firefox",
            "--max-downloads", str(self.max_downloads),
            url
        ]
        
        return command
    
    def _stop_process(self, process):
        """Terminate a subprocess and wait briefly for it to exit"""
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def get_yt_dlp_path(self):
        """Get the correct path to yt-dlp executable"""
        # First check if yt-dlp is in PATH
        path_in_system = shutil.which("yt-dlp")
        if path_in_system:
            return path_in_system
        
        # Fall back to known Python framework location
        yt_dlp_path = "/Library/Frameworks/Python.framework/Versions/3.13/bin/yt-dlp"
        if os.path.exists(yt_dlp_path):
            return yt_dlp_path
        
        return "yt-dlp"  # Default fallback

    def is_yt_dlp_available(self):
        """Check if yt-dlp is available on the system"""
        yt_dlp_path = self.get_yt_dlp_path()
        return yt_dlp_path != "yt-dlp" and os.path.exists(yt_dlp_path)

    def is_network_available(self):
        """Check whether YouTube is reachable"""
        try:
            with socket.create_connection(("www.youtube.com", 443), timeout=3):
                return True
        except OSError:
            return False

    def wait_for_network(self, context):
        """Pause until connectivity is restored or the user stops the job"""
        if self.is_network_available():
            return True

        self.send_message('status', f"No network connection. Waiting to resume {context}...")
        while not self.should_stop.is_set():
            if self.is_network_available():
                self.send_message('status', f"Network restored. Resuming {context}...")
                return True
            time.sleep(2)

        return False

    def is_retriable_network_error(self, error_text):
        """Detect yt-dlp failures that should be retried after connectivity returns"""
        if not error_text:
            return False

        lowered = error_text.lower()
        patterns = [
            "network is unreachable",
            "no route to host",
            "temporary failure in name resolution",
            "name or service not known",
            "failed to resolve",
            "could not resolve host",
            "connection timed out",
            "timed out",
            "connection reset by peer",
            "connection refused",
            "socket timeout",
            "unable to download webpage",
            "unable to download api page",
            "proxyerror",
        ]

        return any(pattern in lowered for pattern in patterns)

    def run_download_process(self, command, folder):
        """Run yt-dlp process and handle output"""
        downloaded = 0
        max_idle = 240  # seconds

        while not self.should_stop.is_set():
            if not self.wait_for_network("the current download"):
                break

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            except FileNotFoundError:
                self.send_message('error', "yt-dlp command not found. Please install yt-dlp and ensure it's in your PATH.")
                break
            except Exception as e:
                self.send_message('error', f"Failed to start yt-dlp: {str(e)}")
                break

            stderr_lines = []
            last_activity = time.time()
            interrupted_by_network = False
            limit_reached = False

            stdout_fd = process.stdout.fileno()
            stderr_fd = process.stderr.fileno()
            streams = {
                stdout_fd: ("stdout", process.stdout),
                stderr_fd: ("stderr", process.stderr),
            }

            while True:
                if self.should_stop.is_set():
                    self._stop_process(process)
                    self.send_message('status', "Download stopped by user")
                    break

                if not self.is_network_available():
                    interrupted_by_network = True
                    self.send_message('status', "Network connection lost. Pausing until it returns...")
                    self._stop_process(process)
                    break

                if process.poll() is not None:
                    break

                ready, _, _ = select.select([stdout_fd, stderr_fd], [], [], 5.0)

                if not ready:
                    idle_time = time.time() - last_activity
                    if not self.is_network_available():
                        interrupted_by_network = True
                        self.send_message('status', "Network connection lost. Pausing until it returns...")
                        self._stop_process(process)
                        break
                    if idle_time > max_idle:
                        self.send_message('log', f"No output for {int(idle_time)}s - terminating")
                        self._stop_process(process)
                        break
                    continue

                for fd in ready:
                    stream_type, stream = streams[fd]
                    line = stream.readline()
                    if not line:
                        continue

                    last_activity = time.time()

                    if stream_type == 'stderr':
                        stderr_lines.append(line)
                        continue

                    try:
                        video_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if video_data.get('_type') == 'playlist':
                        continue

                    self.save_video_metadata(video_data, folder)
                    downloaded += 1

                    progress = (downloaded / self.max_downloads) * 100
                    self.send_message('progress', progress)

                    message = self.format_download_message(video_data)
                    self.send_message('log', message)

                    if downloaded >= self.max_downloads:
                        limit_reached = True
                        self._stop_process(process)
                        break

                if limit_reached:
                    break

            stderr_text = ''.join(stderr_lines)
            returncode = process.poll()

            if self.should_stop.is_set() or limit_reached:
                break

            if interrupted_by_network or self.is_retriable_network_error(stderr_text):
                if not self.wait_for_network("the current download"):
                    break
                continue

            if returncode not in (0, None):
                if downloaded == 0 and stderr_text:
                    self.send_message('error', f"No subtitles downloaded. Error: {stderr_text[:200]}")
                break

            if returncode == 0:
                break

            if downloaded == 0:
                break

        if not self.batch_mode.get():
            if downloaded > 0:
                channel_id = self.extract_channel_id(self.url_input.get().strip())
                self.send_message('status', f"✓ Downloaded {downloaded} subtitles to {channel_id}")
            else:
                self.send_message('status', "No new subtitles found (may already be downloaded)")
    
    # Metadata Functions ⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚
    # note batch functions are AI generated and checked by human
    
    def save_video_metadata(self, video_data, folder):
        """Save video metadata to JSON file"""
        metadata_file = os.path.join(folder, "metadata.json")
        
        # Create metadata entry
        entry = self.create_metadata_entry(video_data)
        
        # Validate entry before saving
        if not entry.get('id'):
            self.send_message('error', "Cannot save metadata: video ID is missing")
            return
        
        # Load existing metadata
        cache_key = os.path.abspath(folder)
        if cache_key in self._metadata_cache:
            metadata = self._metadata_cache[cache_key]
        else:
            metadata = []
            if os.path.exists(metadata_file):
                try:
                    metadata = self.load_json(metadata_file)
                except Exception as e:
                    self.send_message('warning', f"Could not load existing metadata, attempting recovery: {str(e)}")
                    # Try to recover by cleaning corrupted file
                    metadata = self._try_recover_corrupted_metadata(metadata_file)
            self._metadata_cache[cache_key] = metadata
        
        # Add meta if not already present
        existing_ids = self._metadata_id_cache.setdefault(
            cache_key,
            {item.get('id') for item in metadata if isinstance(item, dict) and item.get('id')}
        )
        if entry['id'] not in existing_ids:
            metadata.append(entry)
            existing_ids.add(entry['id'])
        self._metadata_cache[cache_key] = metadata

        self.current_downloads += 1 # increment count of current downloads

    def _flush_metadata_cache(self, folder):
        """Write a channel's accumulated metadata after its download process ends."""
        cache_key = os.path.abspath(folder)
        metadata = self._metadata_cache.pop(cache_key, None)
        self._metadata_id_cache.pop(cache_key, None)
        if metadata is None:
            return

        metadata_file = os.path.join(folder, "metadata.json")
        try:
            self.save_json(metadata_file, metadata)
        except Exception as e:
            self.send_message('error', f"Failed to save metadata: {str(e)}")
    
    def create_metadata_entry(self, video_data, save_timestamp=None):
        """Create a metadata entry from video data with validation"""
        channel_name = video_data.get('channel') or video_data.get('uploader') or ''
        
        # Validate and sanitize values to prevent incomplete/None values
        entry = { # get a lot (a lot) of information about the video, use simple dictionary, this helps in filtering later on
            'id': self._sanitize_value(video_data.get('id', ''), str),
            'title': self._sanitize_value(video_data.get('title', ''), str),
            'url': self._sanitize_value(video_data.get('webpage_url', ''), str),
            'upload_date': self._sanitize_value(video_data.get('upload_date', ''), str),
            'duration': self._sanitize_value(video_data.get('duration', 0), int),
            'view_count': self._sanitize_value(video_data.get('view_count', 0), int),
            'like_count': self._sanitize_value(video_data.get('like_count', 0), int),
            'comment_count': self._sanitize_value(video_data.get('comment_count', 0), int),
            'was_live': self._sanitize_value(video_data.get('was_live', False), bool),
            'is_live': self._sanitize_value(video_data.get('is_live', False), bool),
            'timestamp': datetime.now().isoformat(),
            'channel_name': self._sanitize_value(channel_name, str),
            'channel_id': self._sanitize_value(video_data.get('channel_id', ''), str),
            'channel_url': self._sanitize_value(video_data.get('channel_url', ''), str),
            'subscriber_count': self._sanitize_value(video_data.get('channel_follower_count', 0), int),
            'category': self._sanitize_value(video_data.get('categories', [''])[0] if video_data.get('categories') else '', str)
        }
        
        # Save original timestamp if provided (know when updating metadata to help update it when it is outdated)
        if save_timestamp:
            entry['original_timestamp'] = self._sanitize_value(save_timestamp, str)
        
        return entry
    
    def update_metadata(self, folder):
        """Update metadata for videos that haven't been updated recently"""
        metadata_file = os.path.join(folder, "metadata.json")
        
        if not os.path.exists(metadata_file):
            self.send_message('error', f"No previous downloads found. Directory: {folder}")
            return
        
        # Find videos that need updating
        videos_to_update = self.find_outdated_videos(metadata_file)
        
        if not videos_to_update:
            self.send_message('status', "All metadata is up to date (updated within 7 days)")
            return
        
        self.send_message('status', f"{len(videos_to_update)} videos need metadata updates")
        
        # Sort videos by upload date (newest first) - use upload_date from metadata
        def parse_upload_date(item):
            try:
                upload_date = item.get('upload_date', '')
                if upload_date and len(upload_date) == 8:
                    return datetime.strptime(upload_date, '%Y%m%d')
            except:
                pass
            return datetime.min
        
        videos_to_update.sort(key=parse_upload_date, reverse=True)
        
        # Update in batches
        self.update_videos_metadata(videos_to_update, folder)
    
    def find_outdated_videos(self, metadata_file):
        """Find videos with outdated metadata (older than 7 days)"""
        try:
            metadata = self.load_json(metadata_file)
            outdated = []
            
            for item in metadata:
                if 'id' not in item:
                    continue
                
                timestamp = item.get('timestamp', '')
                if self.is_timestamp_old(timestamp, days=7):
                    outdated.append({
                        'id': item['id'],
                        'title': item.get('title', 'Unknown'),
                        'timestamp': timestamp,
                        'upload_date': item.get('upload_date', '')
                    })
            
            return outdated
        
        except Exception as e:
            self.send_message('error', f"Error reading metadata: {str(e)}")
            return []
    
    def is_timestamp_old(self, timestamp_str, days=7):
        """Check if timestamp is older than specified days"""
        try: # use datetime to make code simpler anfd efficient
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=None) # make naive to be used for comparison
            
            age = datetime.now() - timestamp
            return age > timedelta(days=days)
        
        except (ValueError, TypeError, AttributeError):
            return True  # Treat invalid timestamps as old as errors still need fixing
    
    def update_videos_metadata(self, videos, folder):
        """Update video metadata"""
        # Check if yt-dlp is available
        if not self.is_yt_dlp_available():
            self.send_message('error', "yt-dlp is not installed or not found in PATH. Please install yt-dlp first.")
            return 0

        updated = 0
        total = len(videos)

        list = videos[0:total]
        updated += self.update_batch(list, folder)

        if updated > 0:
            self.send_message('status', f"✓ Updated metadata for {updated}/{total} videos")
        else:
            self.send_message('status', "No metadata was updated")
    
    def update_batch(self, batch, folder):
        """Update a batch of videos"""
        # Build URLs for batch
        urls = [f"https://www.youtube.com/watch?v={v['id']}" for v in batch]

        # Build yt-dlp command
        command = [
            self.get_yt_dlp_path(),
            "--skip-download",
            "--print-json",
            "--no-warnings",
            "--extractor-args", "youtube:player-client=default,mweb"
        ] + urls

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except FileNotFoundError:
            self.send_message('error', "yt-dlp command not found. Please install yt-dlp and ensure it's in your PATH.")
            return 0
        except Exception as e:
            self.send_message('error', f"Failed to start yt-dlp: {str(e)}")
            return 0

        updated = 0

        for line in process.stdout:
            if self.should_stop.is_set():
                process.terminate()
                break
            
            try:
                video_data = json.loads(line)
                
                if video_data.get('_type') == 'playlist':
                    continue
                
                # Find original video info
                video_id = video_data.get('id')
                original = next((v for v in batch if v['id'] == video_id), None)
                
                if original:
                    # Create entry preserving original timestamp
                    entry = self.create_metadata_entry(
                        video_data,
                        save_timestamp=original['timestamp']
                    )
                    
                    # Update in file
                    self.update_metadata_entry(folder, entry)
                    updated += 1
                    
                    # Update progress
                    progress = (updated / len(batch)) * 100
                    self.send_message('progress', progress)
                    
                    # Log update
                    days_old = self.calculate_days_since(original['timestamp'])
                    message = self.format_update_message(entry, days_old)
                    self.send_message('log', message)
            
            except json.JSONDecodeError:
                continue
        
        return updated
    
    def update_metadata_entry(self, folder, entry):
        """Update a single metadata entry in the file"""
        metadata_file = os.path.join(folder, "metadata.json")
        
        try:
            metadata = self.load_json(metadata_file)
            
            # Find and update the entries
            for i, item in enumerate(metadata):
                if item.get('id') == entry['id']:
                    # Keep original timestamp, for future filtering use
                    if 'original_timestamp' not in item and 'timestamp' in item: # usually older downloads
                        entry['original_timestamp'] = item['timestamp']
                    elif 'original_timestamp' in item: 
                        entry['original_timestamp'] = item['original_timestamp']
                    
                    metadata[i] = entry
                    break
            
            self.save_json(metadata_file, metadata)
        
        except Exception as e:
            self.send_message('error', f"Could not update metadata: {str(e)}")
    
    # Consistency Check Functions ⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚
    # Most of these are AI generated and human checked for accuracy
    
    def check_channel(self, folder, channel_name):
        """Check consistency for a single channel"""
        issues_found = 0
        issues_fixed = 0
        
        self.send_message('log', f"\n=== Checking {channel_name} ===")
        
        # Load all data
        metadata_ids, metadata = self.load_metadata_ids(folder)
        archive_ids, duplicate_archive_count = self.load_archive_ids(folder)
        vtx_ids = self.load_vtx_ids(folder)

        # Check 1: Remove duplicate archive entries first (safe, non-destructive)
        if duplicate_archive_count:
            issues_found += duplicate_archive_count
            self.send_message('log', f"  Issue: Found {duplicate_archive_count} duplicate archive entries")
            removed_duplicates = self.remove_duplicate_archive_entries(folder)
            if removed_duplicates:
                issues_fixed += removed_duplicates
                archive_ids, _ = self.load_archive_ids(folder)
        
        # Check 2: Remove archive entries without metadata
        orphaned_archive = []
        for video_id in archive_ids:
            if video_id not in metadata_ids:
                orphaned_archive.append(video_id)
                self.send_message('log', f"  Issue: Archive entry without metadata: {video_id}")
        
        if orphaned_archive:
            issues_found += len(orphaned_archive)
            removed_archive_lines = self.remove_archive_entries(folder, orphaned_archive)
            if removed_archive_lines:
                issues_fixed += removed_archive_lines
                archive_ids, _ = self.load_archive_ids(folder)

        # Check 3: Create archive entries for metadata that have VTX files
        missing_archive = []
        for entry in metadata:
            video_id = entry.get('id')
            if not video_id:
                continue
            
            # If metadata has a VTX file but no archive entry, add to archive
            if video_id in vtx_ids and video_id not in archive_ids:
                missing_archive.append(video_id)
                title = entry.get('title', video_id)
                self.send_message('log', f"  Issue: Metadata with VTX but no archive entry: {title}")
        
        if missing_archive:
            issues_found += len(missing_archive)
            added_archive_lines = self.add_archive_entries(folder, missing_archive)
            if added_archive_lines:
                issues_fixed += added_archive_lines
                archive_ids, _ = self.load_archive_ids(folder)
        
        # Check 4: Remove metadata entries only if they have NO VTX and NO archive
        orphaned_metadata = []
        for entry in metadata:
            video_id = entry.get('id')
            if not video_id:
                continue
            
            # Only remove metadata if it has neither VTX nor archive
            if video_id not in vtx_ids and video_id not in archive_ids:
                orphaned_metadata.append(video_id)
                title = entry.get('title', video_id)
                self.send_message('log', f"  Issue: Metadata without archive or VTX: {title}")
        
        if orphaned_metadata:
            issues_found += len(orphaned_metadata)
            if self.remove_metadata_entries(folder, metadata, orphaned_metadata):
                issues_fixed += len(orphaned_metadata)
        
        # Check 5: Report missing VTX files (informational)
        missing_vtx = metadata_ids & archive_ids - vtx_ids
        if missing_vtx:
            self.send_message('log', 
                f"  Info: {len(missing_vtx)} videos have metadata/archive but no VTX")
        
        if issues_found == 0:
            self.send_message('log', "  ✓ No issues found")
        
        return issues_found, issues_fixed
    
    def load_metadata_ids(self, folder):
        """Load video IDs from metadata.json"""
        metadata_file = os.path.join(folder, "metadata.json")
        
        if not os.path.exists(metadata_file):
            return set(), []
        
        try:
            metadata = self.load_json(metadata_file)
            ids = {item['id'] for item in metadata if 'id' in item}
            return ids, metadata
        except Exception as e:
            self.send_message('log', f"  Error reading metadata: {str(e)}")
            return set(), []
    
    def load_archive_ids(self, folder):
        """Load video IDs from archive.txt"""
        archive_file = os.path.join(folder, "archive.txt")
        
        if not os.path.exists(archive_file):
            return set(), 0
        
        try:
            ids = set()
            duplicate_count = 0
            with open(archive_file, 'r', encoding='utf-8') as f:
                for line in f:
                    video_id = self._parse_archive_video_id(line)
                    if not video_id:
                        continue

                    if video_id in ids:
                        duplicate_count += 1
                    else:
                        ids.add(video_id)

            return ids, duplicate_count
        except Exception as e:
            self.send_message('log', f"  Error reading archive: {str(e)}")
            return set(), 0
    
    def load_vtx_ids(self, folder):
        """Load video IDs from VTX files"""
        vtx_folder = os.path.join(folder, "vtx_files")
        
        if not os.path.exists(vtx_folder):
            return set()
        
        ids = set()
        for filename in os.listdir(vtx_folder):
            if filename.endswith('.vtx'):
                # Extract video ID from filename like [VIDEO_ID].vtx
                match = re.search(r'\[([^\]]+)\]\.vtx', filename)
                if match:
                    ids.add(match.group(1))
        
        return ids
    
    def remove_metadata_entries(self, folder, metadata, video_ids):
        """Remove metadata entries for given video IDs"""
        try:
            metadata_file = os.path.join(folder, "metadata.json")
            cleaned = [m for m in metadata if m.get('id') not in video_ids]
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned, f, indent=2)
            
            self.send_message('log', f"  Fixed: Removed {len(video_ids)} orphaned metadata entries")
            return True
        
        except Exception as e:
            self.send_message('log', f"  Error saving metadata: {str(e)}")
            return False
    
    def remove_archive_entries(self, folder, video_ids):
        """Remove archive entries for given video IDs"""
        archive_file = os.path.join(folder, "archive.txt")
        
        if not os.path.exists(archive_file):
            return 0
        
        try:
            with open(archive_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Keep lines that do not match the target IDs exactly
            cleaned = []
            removed_count = 0
            for line in lines:
                parsed_id = self._parse_archive_video_id(line)
                if parsed_id and parsed_id in video_ids:
                    removed_count += 1
                else:
                    cleaned.append(line)
            
            with open(archive_file, 'w', encoding='utf-8') as f:
                f.writelines(cleaned)
            
            self.send_message('log', f"  Fixed: Removed {removed_count} orphaned archive entries")
            return removed_count
        
        except Exception as e:
            self.send_message('log', f"  Error updating archive: {str(e)}")
            return 0
    
    def add_archive_entries(self, folder, video_ids):
        """Add archive entries for given video IDs"""
        archive_file = os.path.join(folder, "archive.txt")
        
        try:
            # Create archive file if it doesn't exist
            if not os.path.exists(archive_file):
                with open(archive_file, 'w', encoding='utf-8') as f:
                    pass  # Create empty file
            
            # Read existing entries to avoid duplicates
            existing_ids = set()
            if os.path.exists(archive_file):
                with open(archive_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parsed_id = self._parse_archive_video_id(line)
                        if parsed_id:
                            existing_ids.add(parsed_id)
            
            # Append new entries
            added_count = 0
            with open(archive_file, 'a', encoding='utf-8') as f:
                for video_id in video_ids:
                    if video_id not in existing_ids:
                        f.write(f"youtube {video_id}\n")
                        added_count += 1
            
            self.send_message('log', f"  Fixed: Added {added_count} missing archive entries")
            return added_count
        
        except Exception as e:
            self.send_message('log', f"  Error adding archive entries: {str(e)}")
            return 0

    def remove_duplicate_archive_entries(self, folder):
        """Remove duplicated archive lines while keeping first occurrence of each video ID"""
        archive_file = os.path.join(folder, "archive.txt")

        if not os.path.exists(archive_file):
            return 0

        try:
            with open(archive_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            cleaned = []
            seen_ids = set()
            removed_count = 0

            for line in lines:
                parsed_id = self._parse_archive_video_id(line)
                if not parsed_id:
                    cleaned.append(line)
                    continue

                if parsed_id in seen_ids:
                    removed_count += 1
                    continue

                seen_ids.add(parsed_id)
                cleaned.append(line)

            if removed_count:
                with open(archive_file, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned)
                self.send_message('log', f"  Fixed: Removed {removed_count} duplicate archive entries")

            return removed_count

        except Exception as e:
            self.send_message('log', f"  Error deduplicating archive: {str(e)}")
            return 0

    def _parse_archive_video_id(self, line):
        """Extract video ID from a yt-dlp archive line like: youtube VIDEO_ID"""
        stripped = line.strip()
        if not stripped:
            return None

        parts = stripped.split()
        if len(parts) >= 2 and parts[0].lower() == 'youtube':
            return parts[1]

        return None
    
    # Helper Functions ⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚
    # note that this doesn't use any external libraries for formatting
    # this is so the program can be separated from the rest of the codebase if needed
    
    def get_input_folder(self):
        """Get the path to the input directory"""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(script_dir, "data", "input")
    
    def get_channel_folder(self, channel_id):
        """Get the folder path for a channel"""
        return os.path.join(self.get_input_folder(), channel_id)
    
    def get_all_channels(self, input_folder):
        """Get list of all channel directories"""
        return [d for d in os.listdir(input_folder)
                if os.path.isdir(os.path.join(input_folder, d))]
    
    def extract_channel_id(self, url):
        """Extract channel or playlist ID from YouTube URL"""
        # Try different URL patterns
        patterns = [
            r'youtube\.com/c/([^/]+)',
            r'youtube\.com/channel/([^/]+)',
            r'youtube\.com/@([^/]+)',
            r'youtube\.com/user/([^/]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Check for playlist
        playlist_match = re.search(r'[&?]list=([^&]+)', url)
        if playlist_match:
            return playlist_match.group(1)
        
        return "youtube_subtitles"
    
    def load_json(self, filepath):
        """Load JSON data from file with error recovery"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If it's a list, deduplicate by ID
            if isinstance(data, list):
                data = self._deduplicate_entries(data)
            
            return data
        except json.JSONDecodeError as e:
            # Try to recover by loading backup if it exists
            backup_filepath = filepath + '.backup'
            if os.path.exists(backup_filepath):
                self.send_message('warning', f"Corrupted JSON detected, recovering from backup")
                with open(backup_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        data = self._deduplicate_entries(data)
                    return data
            raise e
    
    def _try_recover_corrupted_metadata(self, filepath):
        """Try to recover valid entries from corrupted metadata file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to extract valid JSON objects from the file
            valid_entries = []
            # Look for patterns like {"id": "xxxxx", ...}
            import re
            
            # First, try to clean up obvious corruption patterns
            # Remove orphaned commas (lines that are just commas or whitespace+comma)
            cleaned = re.sub(r'^\s*,\s*$', '', content, flags=re.MULTILINE)
            
            # Fix missing opening braces: pattern like },     "id" should be },\n    {
            # This occurs when second entry is missing its opening brace
            cleaned = re.sub(r'\},\s*"id"', r'},\n    {\n        "id"', cleaned)
            
            # Fix malformed field names: "channel_id: "value" should be "channel_id": "value"
            # Pattern: quote-field name-colon-space-quote instead of quote-field-quote-colon-space-quote
            cleaned = re.sub(r'"([a-z_]+):\s*"', r'"\1": "', cleaned)
            
            # Fix empty values with orphaned commas: "field": , should be "field": ""
            cleaned = re.sub(r':\s*,', r': "",', cleaned)
            
            # Fix truncated URLs: "url": "http, should be "url": ""
            cleaned = re.sub(r':\s*"(https?[^"]*)\s*,', r': "",', cleaned)
            
            # Try parsing the cleaned content as JSON array first
            try:
                data = json.loads(cleaned)
                if isinstance(data, list):
                    # Deduplicate by ID
                    data = self._deduplicate_entries(data)
                    if data:
                        self.send_message('status', f"Successfully recovered {len(data)} entries after cleaning corruption patterns")
                        return data
            except json.JSONDecodeError:
                pass
            
            # Try to fix unclosed strings (e.g., "timestamp": "2025-07-29 without closing quote)
            # Add closing quotes to incomplete string values
            cleaned = re.sub(r':\s*"([^"]*?)\s*\n\s*"([a-z_]+)"\s*:', r': "\1",\n            "\2": ', cleaned)
            
            try:
                data = json.loads(cleaned)
                if isinstance(data, list):
                    data = self._deduplicate_entries(data)
                    if data:
                        self.send_message('status', f"Successfully recovered {len(data)} entries after fixing unclosed strings")
                        return data
            except json.JSONDecodeError:
                pass
            
            # If that doesn't work, try to extract individual JSON objects
            # Split by top-level objects - look for { to } pairs
            # This handles cases where entries are malformed but contain valid subsets
            pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(pattern, content)
            
            for match in matches:
                try:
                    # Try to fix unclosed strings in this match
                    fixed_match = re.sub(r':\s*"([^"]*?)\s*\n', r': "\1",\n', match)
                    
                    # Fix empty values in this match
                    fixed_match = re.sub(r':\s*,', r': "",', fixed_match)
                    
                    # Fix malformed keys in this match
                    fixed_match = re.sub(r'"([a-z_]+):\s*"', r'"\1": "', fixed_match)
                    
                    entry = json.loads(fixed_match)
                    if isinstance(entry, dict) and 'id' in entry and entry['id']:
                        valid_entries.append(entry)
                except (json.JSONDecodeError, ValueError):
                    # Skip malformed objects
                    pass
            
            if valid_entries:
                # Deduplicate recovered entries
                valid_entries = self._deduplicate_entries(valid_entries)
                self.send_message('status', f"Recovered {len(valid_entries)} valid entries from corrupted metadata")
                return valid_entries
            
            # If extraction failed, try backup
            backup_filepath = filepath + '.backup'
            if os.path.exists(backup_filepath):
                try:
                    with open(backup_filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            data = self._deduplicate_entries(data)
                        return data
                except:
                    pass
            
            return []
        except Exception as e:
            self.send_message('error', f"Could not recover metadata: {str(e)}")
            return []
    
    def save_json(self, filepath, data):
        """Save data to JSON file using atomic write to prevent corruption"""
        # Validate data before saving
        self._validate_json_data(data)
        
        # Create backup of existing file before overwriting
        if os.path.exists(filepath):
            try:
                os.replace(filepath, filepath + '.backup')
            except:
                pass
        
        # Use atomic write: write to temp file first, then rename
        temp_filepath = filepath + '.tmp'
        try:
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # Verify the JSON is valid before committing
            with open(temp_filepath, 'r', encoding='utf-8') as f:
                json.load(f)  # This will raise if JSON is invalid
            
            # Atomic rename
            os.replace(temp_filepath, filepath)
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            raise e
    
    def _is_truncated_value(self, val_str):
        """Check if a string value appears truncated"""
        # Truncation indicators: ends with comma, colon, or quote
        return val_str.endswith((',', ':', '"'))
    
    def _has_field_corruption(self, val_str):
        """Check if a string contains evidence of field corruption"""
        # Multiple colons suggest merged fields
        if val_str.count(':') > 2:
            return True
        # Multiple quotes with colons suggest corrupted JSON syntax
        if val_str.count('"') > 2 and ':' in val_str:
            return True
        return False
    
    def _extract_duplicated_content(self, val_str):
        """Extract clean content from a duplicated/merged value"""
        if ': "' not in val_str:
            return None
        
        parts = val_str.split(': "', 1)
        if len(parts) != 2:
            return None
        
        prefix, rest = parts[0].strip(), parts[1]
        # If prefix is substantial and appears in rest, content was duplicated
        if prefix and len(prefix) > 10:
            prefix_words = prefix.split()
            if prefix_words:
                first_words = ' '.join(prefix_words[:min(3, len(prefix_words))])
                if first_words and rest.startswith(first_words):
                    # Duplicated content found - use rest version
                    return rest
        return None
    
    def _validate_url(self, val_str):
        """Validate a URL value"""
        if not val_str.startswith('http'):
            return val_str
        # URLs should end properly, not with punctuation
        if self._is_truncated_value(val_str):
            return ''
        # YouTube URLs have minimum length
        if len(val_str) < 25:
            return ''
        # Must be YouTube domain
        if 'youtube.com/watch' not in val_str and 'youtube.com/channel' not in val_str:
            return ''
        return val_str
    
    def _validate_date(self, val_str):
        """Validate a date value (YYYYMMDD or YYYY-MM-DD format)"""
        if not (val_str.isdigit() or '-' in val_str):
            return val_str
        if val_str.replace('-', '').isdigit():
            date_digits = val_str.replace('-', '')
            # Must be at least 8 digits (YYYYMMDD)
            if len(date_digits) < 8 or self._is_truncated_value(val_str):
                return ''
            return val_str
        return val_str
    
    def _validate_channel_id(self, val_str):
        """Validate a channel ID (UC + 24 chars)"""
        if not val_str.startswith('UC'):
            return val_str
        if len(val_str) < 24 or self._is_truncated_value(val_str):
            return ''
        return val_str
    
    def _validate_video_id(self, val_str):
        """Validate a video ID (11 alphanumeric characters)"""
        if len(val_str) != 11:
            return val_str
        if val_str.replace('-', '').replace('_', '').isalnum():
            return val_str
        return ''
    
    def _sanitize_value(self, value, expected_type):
        """Sanitize a value to ensure it's valid and complete for JSON serialization"""
        # Handle None values
        if value is None:
            return {str: '', int: 0, bool: False}.get(expected_type, value)
        
        try:
            if expected_type == str:
                val_str = str(value).strip()
                
                # Check for severely malformed values (starts with special chars)
                if val_str.startswith((':',  ',', '"', '|')):
                    return ''
                
                # Check for field corruption patterns
                if self._has_field_corruption(val_str):
                    return ''
                
                # Try to extract clean content from duplicated values
                cleaned = self._extract_duplicated_content(val_str)
                if cleaned:
                    return cleaned
                
                # Type-specific validation
                if val_str.startswith('http'):
                    return self._validate_url(val_str)
                if val_str and (val_str.isdigit() or '-' in val_str):
                    return self._validate_date(val_str)
                if val_str.startswith('UC'):
                    return self._validate_channel_id(val_str)
                if len(val_str) == 11:
                    return self._validate_video_id(val_str)
                
                # General string validation
                if self._is_truncated_value(val_str) or len(val_str) < 3:
                    return ''
                if '\\u' in repr(val_str) and len(val_str) < 10:
                    return ''
                
                return val_str
            elif expected_type == int:
                return int(value) if value else 0
            elif expected_type == bool:
                return bool(value)
            return value
        except (ValueError, TypeError):
            return {str: '', int: 0, bool: False}.get(expected_type, value)
    
    CRITICAL_FIELDS = {
        'id': '', 'title': '', 'url': '', 'upload_date': '',
        'duration': 0, 'view_count': 0, 'like_count': 0, 'comment_count': 0,
        'was_live': False, 'is_live': False, 'timestamp': '',
        'channel_name': '', 'channel_id': '', 'channel_url': '', 'subscriber_count': 0
    }
    
    def _validate_json_data(self, data):
        """Validate JSON data structure before saving to prevent corruption"""
        if not isinstance(data, list):
            return
        
        cleaned_data = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            
            # Check required fields
            if not item.get('id'):
                self.send_message('warning', f"Entry {i} has no ID, skipping")
                continue
            
            # Check for severe corruption
            if self._is_entry_severely_corrupted(item):
                self.send_message('warning', f"Entry {i} (ID: {item.get('id')}) is severely corrupted, skipping")
                continue
            
            # Clean the entry
            cleaned_item = self._clean_entry_fields(item)
            if cleaned_item.get('id'):
                cleaned_data.append(cleaned_item)
            else:
                self.send_message('warning', f"Entry {i} lost required ID during cleaning, skipping")
        
        data.clear()
        data.extend(cleaned_data)
    
    def _clean_entry_fields(self, item):
        """Remove duplicate keys and invalid values from an entry"""
        cleaned_item = {}
        seen_keys = set()
        
        # Process existing fields, skip duplicates
        for key, value in item.items():
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cleaned_item[key] = self._clean_field_value(key, value)
        
        # Ensure all critical fields exist with defaults
        for field, default_val in self.CRITICAL_FIELDS.items():
            if field not in cleaned_item:
                cleaned_item[field] = default_val
        
        return cleaned_item
    
    def _clean_field_value(self, key, value):
        """Clean a single field value"""
        if value is None:
            return ''
        
        if isinstance(value, str):
            # Check for truncated strings
            if value.endswith((',', ':', '"')):
                return ''
            # Check for incomplete URLs
            if value.startswith('http') and len(value) < 20:
                return ''
            # Check for incomplete channel IDs
            if value.startswith('UC') and len(value) < 24:
                return ''
            return value
        
        if not isinstance(value, (int, float, bool)):
            # Try to convert unexpected types
            try:
                str_val = str(value)
                return '' if str_val.endswith((',', ':')) else str_val
            except:
                return ''
        
        return value
    
    def _is_entry_severely_corrupted(self, item):
        """Check if an entry is severely corrupted (fields merged/misplaced)"""
        if not isinstance(item, dict):
            return True
        
        # Check ID field for corruption
        id_value = item.get('id', '')
        if isinstance(id_value, str):
            if id_value.startswith((':',  ',', '"')):
                return True
            if id_value.count(':') > 1 or len(id_value) > 200 or '{' in id_value:
                return True
            if id_value.endswith((':', 'GROU:', 'PLAYGRO:', 'PLAYGROUND')):
                if not (len(id_value) == 11 and id_value.replace('-', '').replace('_', '').isalnum()):
                    return True
        
        # Check for corrupted field structures
        suspicious_keys = self._count_suspicious_keys(item)
        if suspicious_keys >= 2:
            return True
        
        # Check title field for corruption patterns
        title_value = item.get('title', '')
        if isinstance(title_value, str):
            if '": "' in title_value or title_value.startswith('"') or title_value.endswith('":'):
                return True
            if self._has_duplicated_content(title_value):
                return True
        
        # Check for field values containing merged field data
        if self._has_merged_field_data(item):
            return True
        
        return False
    
    def _count_suspicious_keys(self, item):
        """Count keys that look like corrupted/truncated field names"""
        truncated_fields = ('uplo', 'vie', 'tim', 'cha', 'sub', 'com', 'lik', 'was')
        suspicious_count = 0
        
        for key in item.keys():
            if not isinstance(key, str):
                return float('inf')  # Non-string key is definitely corrupted
            if '"' in key or "'" in key or key.startswith(':'):
                return float('inf')  # Corrupted key
            if key in truncated_fields:
                suspicious_count += 1
        
        return suspicious_count
    
    def _has_duplicated_content(self, val_str):
        """Check if a value has duplicated/repeated content pattern"""
        if ': "' not in val_str:
            return False
        
        parts = val_str.split(': "', 1)
        if len(parts) != 2:
            return False
        
        prefix, rest = parts[0], parts[1]
        if not prefix or len(prefix) <= 10:
            return False
        
        # Check if prefix words appear at start of rest (duplication pattern)
        prefix_words = ' '.join(prefix.split()[:3])
        return bool(prefix_words and prefix_words in rest and len(prefix_words) > 10)
    
    def _has_merged_field_data(self, item):
        """Check if entry contains field values with merged JSON data"""
        for key, value in item.items():
            if isinstance(value, str):
                # Pattern: ": " in long strings indicates merged field data
                if ': "' in value and len(value) > 50:
                    return True
        return False
    
    def _deduplicate_entries(self, data):
        """Remove duplicate entries from metadata list (keep first occurrence of each ID)"""
        if not isinstance(data, list):
            return data
        
        seen_ids = set()
        duplicates_removed = 0
        deduplicated = []
        
        for item in data:
            if isinstance(item, dict):
                video_id = item.get('id')
                if video_id:
                    if video_id in seen_ids:
                        duplicates_removed += 1
                        continue  # Skip this duplicate
                    seen_ids.add(video_id)
            
            deduplicated.append(item)
        
        if duplicates_removed > 0:
            self.send_message('status', f"Removed {duplicates_removed} duplicate entries from metadata")
        
        return deduplicated
    
    # Formatting Functions ⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚
    # note that this doesn't use any external libraries for formatting
    
    def format_download_message(self, video_data):
        """Format a message for a downloaded video to be displayed in the log"""
        title = video_data.get('title', 'Unknown')
        upload_date = video_data.get('upload_date', '')
        view_count = video_data.get('view_count', 0)
        category = video_data.get('categories', [''])[0] if video_data.get('categories') else ''
        channel = video_data.get('channel') or video_data.get('uploader', '')
        
        # Build status tags
        status = [] # these help in identifying live videos in the log for user to see and determine what they want to do with them
        if video_data.get('is_live'): 
            status.append("LIVE")
        elif video_data.get('was_live'):
            status.append("Was Live")
        
        status_str = f" [{', '.join(status)}]" if status else ""
        
        # Format date
        date_str = self.format_date(upload_date)
        
        # Build message
        message = f"Downloaded: {title}{status_str} ({date_str})"
        
        if view_count:
            message += f" | Views: {self.format_number(view_count)}"
        if category:
            message += f" | Category: {category}"
        if channel:
            message += f" | Channel: {channel}"
        
        return message # information about the video!
    
    def format_update_message(self, video_data, days_old):
        """Format a message for an updated video"""
        title = video_data.get('title', 'Unknown')
        view_count = video_data.get('view_count', 0)
        category = video_data.get('category', '')
        channel = video_data.get('channel_name', '')
        
        # Build status tags
        status = []
        if video_data.get('is_live'):
            status.append("LIVE")
        elif video_data.get('was_live'):
            status.append("Was Live")
        
        status_str = f" [{', '.join(status)}]" if status else ""
        
        # Build message
        message = f"Updated: {title}{status_str} ({days_old} days old)"
        
        if view_count:
            message += f" | Views: {self.format_number(view_count)}"
        if category:
            message += f" | Category: {category}"
        if channel:
            message += f" | Channel: {channel}"
        
        return message # information about the updated video! (simpler and different from download message)
    
    def format_date(self, date_str):
        """Format date string from YYYYMMDD to YYYY-MM-DD"""
        if not date_str or len(date_str) != 8:
            return date_str
        
        try:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except:
            return date_str
    
    def format_number(self, number):
        """Format number with K, M, B suffixes"""
        if not number:
            return "0"
        
        number = int(number)
        
        if number >= 1_000_000_000:
            return f"{number / 1_000_000_000:.1f}B"
        elif number >= 1_000_000:
            return f"{number / 1_000_000:.1f}M"
        elif number >= 1_000:
            return f"{number / 1_000:.1f}K"
        else:
            return str(number)
    
    def calculate_days_since(self, timestamp_str):
        """Calculate days since a timestamp"""
        try: # use datetime for efficiency and accuracy
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=None)
            
            days = (datetime.now() - timestamp).days
            return str(days)
        except (ValueError, TypeError, AttributeError):
            return "?"
    
    # Message Queue Functions ⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚⋆.˚𖦹⋆✮⋆.˚
    
    def send_message(self, msg_type, content):
        """Small helper: Send a message from worker thread to UI thread"""
        self.messages.put((msg_type, content))
    
    def check_messages(self):
        """Check for messages from worker thread and handle them"""
        try:
            while True:
                msg_type, content = self.messages.get_nowait()
                self.handle_message(msg_type, content)
        except queue.Empty:
            pass
        
        # Check again in 100ms
        self.after(100, self.check_messages)
    
    def handle_message(self, msg_type, content):
        """Handle a message from the worker thread"""
        if msg_type == 'status': # status bar
            self.status_text.set(content)     
        elif msg_type == 'progress': # update progress bar
            self.progress_bar['value'] = content
        
        elif msg_type == 'log': # append to log output
            self.output.insert(tk.END, content + "\n")
            self.output.see(tk.END)
        elif msg_type == 'warning':
            message = f"[WARNING] {content}"
            self.output.insert(tk.END, message + "\n")
            self.output.see(tk.END)
            self.status_text.set(content)
        elif msg_type == 'error':
            message = f"[ERROR] {content}"
            self.output.insert(tk.END, message + "\n")
            self.output.see(tk.END)
            self.status_text.set(content)
        elif msg_type == 'done': # re-enable buttons
            self.enable_buttons()


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    downloader = SubtitleDownloader(root)
    downloader.protocol("WM_DELETE_WINDOW", root.destroy)

    root.mainloop()

    #caffeinate -i python3 core/atlas_downloader.py

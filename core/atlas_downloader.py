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
    from helpers.vtt_helper import VTTSegment
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
        self.batch_download_number = 20

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
        container = ttk.Frame(self, padding="20")
        container.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        container.columnconfigure(0, weight=1)
        container.rowconfigure(8, weight=1)

        self.create_url_input(container)
        self.create_options(container)
        self.create_buttons(container)

        self.style.configure("red.Horizontal.TProgressbar", background='red', foreground='black')

        self.progress_bar = ttk.Progressbar(container, style="red.Horizontal.TProgressbar", mode='determinate')
        self.progress_bar.grid(column=0, row=6, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        self.status_text = tk.StringVar()
        ttk.Label(container, textvariable=self.status_text).grid(
            column=0, row=7, columnspan=3, sticky=tk.W, pady=(10, 0))

        self.create_output_area(container)

    def create_url_input(self, parent):
        ttk.Label(parent, text="YouTube Channel URL or Playlist URL:").grid(
            column=0, row=0, columnspan=3, sticky=tk.W, pady=(0, 5))

        self.url_input = ttk.Entry(parent, width=100)
        self.url_input.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

    def create_options(self, parent):
        self.metadata_only = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Update metadata only (doesn't download new subtitles)",
            variable=self.metadata_only
        ).grid(column=0, row=2, columnspan=3, sticky=tk.W, pady=(5, 10))

        batch_frame = ttk.Frame(parent)
        batch_frame.grid(column=0, row=3, columnspan=3, sticky=tk.W, pady=(5, 10))

        self.batch_mode = tk.BooleanVar()
        ttk.Checkbutton(
            batch_frame,
            text=f"Batch download: {self.batch_download_number} videos from each channel in input directory",
            variable=self.batch_mode
        ).grid(column=0, row=0, sticky=tk.W)

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
        button_frame = ttk.Frame(parent)
        button_frame.grid(column=0, row=4, columnspan=3, sticky=tk.W, pady=(10, 0))

        self.download_button = ttk.Button(
            button_frame,
            text="Download Subtitles",
            command=self.start_download
        )
        self.download_button.grid(column=0, row=0, sticky=tk.W)

        self.consistency_button = ttk.Button(
            button_frame,
            text="Check Consistency",
            command=self.start_consistency_check
        )
        self.consistency_button.grid(column=1, row=0, sticky=tk.W, padx=(20, 0))

    def create_output_area(self, parent):
        output_frame = ttk.Frame(parent)
        output_frame.grid(column=0, row=8, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output = tk.Text(output_frame, height=18, wrap=tk.WORD)
        self.output.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output.yview)
        scrollbar.grid(column=1, row=0, sticky=(tk.N, tk.S))
        self.output['yscrollcommand'] = scrollbar.set

    # Main Control Functions

    def start_download(self):
        if self.is_working():
            messagebox.showwarning("Warning", "Operation already in progress!", parent=self)
            return

        self.should_stop.clear()
        self.prepare_for_download()
        self.worker_thread = threading.Thread(target=self.download_worker, daemon=True)
        self.worker_thread.start()

    def start_consistency_check(self):
        if self.is_working():
            messagebox.showwarning("Warning", "Operation already in progress!", parent=self)
            return

        self.should_stop.clear()
        self.prepare_for_consistency_check()
        self.worker_thread = threading.Thread(target=self.check_consistency_worker, daemon=True)
        self.worker_thread.start()

    def is_working(self):
        return self.worker_thread and self.worker_thread.is_alive()

    def prepare_for_download(self):
        self.disable_buttons()
        self.progress_bar['value'] = 0
        self.output.delete(1.0, tk.END)
        self.current_downloads = 0

        if self.batch_mode.get():
            button_text = "Batch Downloading..."
        elif self.metadata_only.get():
            button_text = "Updating Metadata..."
        else:
            button_text = "Downloading..."

        self.download_button.config(text=button_text)

    def prepare_for_consistency_check(self):
        self.disable_buttons()
        self.consistency_button.config(text="Checking...")
        self.progress_bar['value'] = 0
        self.output.delete(1.0, tk.END)
        self.status_text.set("Starting consistency check...")

    def disable_buttons(self):
        self.download_button.config(state=tk.DISABLED)
        self.consistency_button.config(state=tk.DISABLED)
        self.batch_filter_button.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.download_button.config(state=tk.NORMAL, text="Download Subtitles")
        self.consistency_button.config(state=tk.NORMAL, text="Check Consistency")
        self.batch_filter_button.config(state=tk.NORMAL)

    # Worker Thread Functions

    def download_worker(self):
        try:
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

    # Batch Download Functions

    def process_batch_download(self):
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

        if reverse_order:
            channels.reverse()

        order_label = "Z→0" if reverse_order else "0→Z"
        self.send_message('status', f"Processing {len(channels)} channels ({order_label})")

        for i, channel in enumerate(channels):
            if self.should_stop.is_set():
                break

            folder = os.path.join(input_folder, channel)
            self.process_single_channel_batch(folder, channel, i + 1, len(channels))

            progress = ((i + 1) / len(channels)) * 100
            self.send_message('progress', progress)

        self.send_message('status', f"✓ Batch complete. Processed {len(channels)} channels")

    def load_batch_handle_list(self):
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

        created = 0
        normalized_handles = []
        pattern = re.compile(r'^(?:https?://)?(?:www\.)?youtube\.com/@(?P<handle>[A-Za-z0-9_\-\.]+)/*$', re.IGNORECASE)

        for h in handles:
            m = pattern.match(h.strip())
            if m:
                handle_name = m.group('handle')
                folder = os.path.join(self.get_input_folder(), handle_name)
                try:
                    if not os.path.exists(folder):
                        os.makedirs(folder, exist_ok=True)
                        created += 1

                    metadata_file = os.path.join(folder, 'metadata.json')
                    if not os.path.exists(metadata_file):
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

                    normalized_handles.append(handle_name)
                    continue
                except Exception as e:
                    self.send_message('log', f"Error preparing folder for {h}: {str(e)}")

            normalized_handles.append(h)

        self.batch_filter_handles = normalized_handles
        self.batch_filter_button.config(text=f"Handle List: {len(normalized_handles)}")
        if created:
            self.status_text.set(f"Loaded {len(normalized_handles)} handles ({created} new channels created)")
        else:
            self.status_text.set(f"Loaded {len(normalized_handles)} handles for batch filter")

    def parse_handles_from_text(self, text):
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
        if not metadata:
            return None

        for entry in metadata:
            if isinstance(entry, dict):
                channel_url = entry.get('channel_url', '').strip()
                if channel_url:
                    return channel_url

        for entry in metadata:
            if isinstance(entry, dict):
                channel_id = entry.get('channel_id', '').strip()
                if channel_id:
                    return f"https://www.youtube.com/channel/{channel_id}"

        if channel_name:
            return f"https://www.youtube.com/c/{channel_name}"

        return None

    def _make_minimal_channel_metadata(self, channel_name):
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

            old_limit = self.max_downloads
            self.max_downloads = self.batch_download_number
            try:
                self.download_from_channel(folder, url)
            finally:
                self.max_downloads = old_limit

        except Exception as e:
            self.send_message('log', f"Error processing {channel_name}: {str(e)}")

    def download_from_channel(self, folder, url):
        """Download subtitles from a YouTube channel or playlist."""
        self._prepare_download_state(folder)

        vtt_folder = os.path.join(folder, "vtt_files")
        os.makedirs(vtt_folder, exist_ok=True)

        command = self.build_download_command(vtt_folder, url)

        if not self.batch_mode.get():
            self.send_message('status', f"Starting download (max {self.max_downloads} subtitles)...")

        try:
            self.run_download_process(command, folder)
        finally:
            # Fallback: explicitly fetch metadata for any video whose ID appears
            # in vtt/vtx filenames but is not yet in metadata.json.
            try:
                self._ensure_metadata_for_downloaded(folder)
            except Exception as e:
                self.send_message('error', f"Metadata fetch failed: {str(e)}")

            self._flush_metadata_cache(folder)
            self._finalize_vtx_transcripts(folder)

    def _prepare_download_state(self, folder):
        _, metadata = self.load_metadata_ids(folder)
        metadata_ids = {
            entry['id'] for entry in metadata
            if self._is_video_metadata_entry(entry)
        }
        archive_ids, _ = self.load_archive_ids(folder)

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
        if not isinstance(entry, dict) or not entry.get('id'):
            return False
        if entry.get('upload_date') or entry.get('duration') is not None:
            return True
        return '/watch?' in entry.get('url', '')

    def _finalize_vtx_transcripts(self, folder):
        vtt_dir = os.path.join(folder, "vtt_files")
        vtx_dir = os.path.join(folder, "vtx_files")
        txt_dir = os.path.join(folder, "txt_files")

        if not os.path.exists(vtt_dir):
            return

        os.makedirs(vtx_dir, exist_ok=True)

        converter = VTTSegment()
        result = converter.convert_vtt_directory_to_vtx(vtt_dir, vtx_dir)

        if result['converted']:
            self.send_message('log', f"  Converted {len(result['converted'])} VTT files to VTX")

        if result['failed']:
            self.send_message('log', f"  Warning: {len(result['failed'])} VTT files failed to convert")

        deleted_count = 0
        for vtt_file in result['converted']:
            base_name = os.path.splitext(vtt_file)[0]
            vtx_file = os.path.join(vtx_dir, f"{base_name}.vtx")
            txt_file = os.path.join(txt_dir, f"{base_name}.txt")
            vtt_path = os.path.join(vtt_dir, vtt_file)

            if os.path.exists(vtx_file) and os.path.exists(txt_file):
                try:
                    if os.path.getsize(txt_file) > 0:
                        os.remove(vtt_path)
                        deleted_count += 1
                except Exception as e:
                    self.send_message('log', f"  Warning: Could not delete {vtt_file}: {str(e)}")

        if deleted_count > 0:
            self.send_message('log', f"  Deleted {deleted_count} verified VTT files")

        remaining_vtt = [f for f in os.listdir(vtt_dir) if f.endswith('.vtt')] if os.path.exists(vtt_dir) else []
        if remaining_vtt:
            self.send_message('log', f"  Warning: {len(remaining_vtt)} unconfirmed VTT files remain in vtt_files/")
            self.send_message('log', f"  These will be kept for manual review or retry")

        if os.path.exists(vtt_dir) and not os.listdir(vtt_dir):
            try:
                os.rmdir(vtt_dir)
                self.send_message('log', f"  Removed empty vtt_files directory")
            except Exception:
                pass

    def build_download_command(self, vtt_folder, url):
        """Build the yt-dlp subtitle download command.

        Uses --write-info-json instead of --print-json because modern yt-dlp
        suppresses stdout JSON when combined with --skip-download. The info.json
        files are written to disk and picked up by _scan_new_info_json().
        """
        archive_file = os.path.join(os.path.dirname(vtt_folder), "archive.txt")
        output_template = os.path.join(vtt_folder, "%(title)s [%(id)s].%(ext)s")

        command = [
            self.get_yt_dlp_path(),
            "--write-auto-sub",
            "--sub-lang", "en.*,-en-orig",
            "--skip-download",
            "--extractor-args", "youtube:skip=translated_subs",
            "--convert-subs", "vtt",
            #"--dump-json",
            #"--write-info-json",
            "--download-archive", archive_file,
            "--no-warnings",
            "--force-write-archive",
            "--no-overwrites",
            "--extractor-args", "youtube:player-client=mweb,tv,web_safari",
            "--remote-components", "ejs:github",
            "--cookies-from-browser", "safari",
            "-o", output_template,
            "--max-downloads", str(self.max_downloads),
            url
        ]

        return command

    def _stop_process(self, process):
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def get_yt_dlp_path(self):
        """Get the correct path to yt-dlp executable.

        Prefers pipx-managed installs because they have the bgutil PO-token
        plugin injected. Falls back to PATH and then a plain name.
        """
        candidates = [
            os.path.expanduser("~/Library/Application Support/pipx/venvs/yt-dlp/bin/yt-dlp"),
            os.path.expanduser("~/.local/pipx/venvs/yt-dlp/bin/yt-dlp"),
            os.path.expanduser("~/.local/bin/yt-dlp"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        path_in_system = shutil.which("yt-dlp")
        if path_in_system:
            return path_in_system

        return "yt-dlp"

    def is_yt_dlp_available(self):
        path = self.get_yt_dlp_path()
        if path == "yt-dlp":
            return shutil.which("yt-dlp") is not None
        return os.path.exists(path)

    def is_network_available(self):
        try:
            with socket.create_connection(("www.youtube.com", 443), timeout=3):
                return True
        except OSError:
            return False

    def wait_for_network(self, context):
        if self.is_network_available():
            return True

        self.send_message('status', f"No network connection. Waiting to resume {context}...")
        while not self.should_stop.is_set():
            if self.is_network_available():
                self.send_message('status', f"Network restored. Resuming {context}...")
                return True
            time.sleep(2)

        return False

    def _extract_video_ids_from_subtitle_files(self, folder):
        """Extract video IDs from filenames in vtt_files/ and vtx_files/."""
        ids = set()
        for subfolder in ("vtt_files", "vtx_files"):
            path = os.path.join(folder, subfolder)
            if not os.path.exists(path):
                continue
            try:
                entries = os.listdir(path)
            except OSError:
                continue
            for filename in entries:
                # Subtitle filenames look like 'Title [VIDEO_ID].en.vtt'
                match = re.search(r'\[([A-Za-z0-9_-]{11})\]', filename)
                if match:
                    ids.add(match.group(1))
        return ids


    def _ensure_metadata_for_downloaded(self, folder):
        """Fetch metadata via --dump-json for any downloaded IDs missing from cache."""
        downloaded_ids = self._extract_video_ids_from_subtitle_files(folder)
        if not downloaded_ids:
            return 0

        cache_key = os.path.abspath(folder)
        existing_ids = self._metadata_id_cache.get(cache_key, set())
        missing_ids = downloaded_ids - existing_ids

        if not missing_ids:
            return 0

        self.send_message(
            'log',
            f"  Fetching metadata for {len(missing_ids)} videos via --dump-json..."
        )
        return self._fetch_metadata_for_ids(sorted(missing_ids), folder)


    def _fetch_metadata_for_ids(self, video_ids, folder):
        """Fetch metadata for the given video IDs using --dump-json (stdout JSON).

        This mirrors the pattern used by update_batch, which we know works.
        No --skip-download is needed; --dump-json simulates by itself.
        """
        if not video_ids:
            return 0

        batch_size = 25
        total_added = 0

        for i in range(0, len(video_ids), batch_size):
            if self.should_stop.is_set():
                break

            batch = video_ids[i:i + batch_size]
            urls = [f"https://www.youtube.com/watch?v={vid}" for vid in batch]

            command = [
                self.get_yt_dlp_path(),
                "--dump-json",
                "--no-warnings",
                "--extractor-args", "youtube:player-client=mweb,tv,web_safari",
                "--remote-components", "ejs:github",
            ] + urls

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            except Exception as e:
                self.send_message('error', f"Failed to start metadata fetch: {str(e)}")
                continue

            for line in process.stdout:
                if self.should_stop.is_set():
                    process.terminate()
                    break
                try:
                    video_data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if video_data.get('_type') == 'playlist':
                    continue
                if self.save_video_metadata(video_data, folder):
                    total_added += 1
                    self.send_message('log', self.format_download_message(video_data))

            process.wait()

        return total_added
        
    def is_retriable_network_error(self, error_text):
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

    def _scan_new_info_json(self, vtt_folder, folder, processed_ids):
        """Scan vtt_folder for *.info.json files we haven't processed yet.

        Returns list of video_data dicts that were newly added to metadata.
        Skips partially-written files by requiring a minimum size and valid JSON.
        """
        if not os.path.exists(vtt_folder):
            return []

        new_items = []
        try:
            entries = os.listdir(vtt_folder)
        except OSError:
            return []

        for filename in entries:
            if not filename.endswith('.info.json'):
                continue

            filepath = os.path.join(vtt_folder, filename)

            try:
                if os.path.getsize(filepath) < 100:
                    continue
                with open(filepath, 'r', encoding='utf-8') as f:
                    video_data = json.load(f)
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue

            if not isinstance(video_data, dict):
                continue

            video_id = video_data.get('id')
            if not video_id or video_id in processed_ids:
                continue

            processed_ids.add(video_id)
            if self.save_video_metadata(video_data, folder):
                new_items.append(video_data)

        return new_items

    def run_download_process(self, command, folder):
        """Run yt-dlp subtitle process and detect new .info.json files.

        Instead of parsing stdout JSON (which modern yt-dlp suppresses when
        --skip-download is present), we poll the vtt_files folder for new
        *.info.json files that yt-dlp writes via --write-info-json.
        """
        downloaded = 0
        max_idle = 240
        vtt_folder = os.path.join(folder, "vtt_files")
        processed_ids = set()

        while not self.should_stop.is_set():
            if not self.wait_for_network("the current download"):
                break

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
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

            stderr_fd = process.stderr.fileno()

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
                    new_items = self._scan_new_info_json(vtt_folder, folder, processed_ids)
                    if new_items:
                        last_activity = time.time()
                        downloaded += len(new_items)
                        for video_data in new_items:
                            self.send_message('log', self.format_download_message(video_data))
                        progress = min((downloaded / self.max_downloads) * 100, 100)
                        self.send_message('progress', progress)
                    break

                try:
                    ready, _, _ = select.select([stderr_fd], [], [], 2.0)
                except (OSError, ValueError):
                    ready = []

                if ready:
                    try:
                        line = process.stderr.readline()
                    except Exception:
                        line = ''
                    if line:
                        stderr_lines.append(line)
                        last_activity = time.time()

                new_items = self._scan_new_info_json(vtt_folder, folder, processed_ids)
                if new_items:
                    last_activity = time.time()
                    downloaded += len(new_items)
                    for video_data in new_items:
                        self.send_message('log', self.format_download_message(video_data))
                    progress = min((downloaded / self.max_downloads) * 100, 100)
                    self.send_message('progress', progress)

                if time.time() - last_activity > max_idle and process.poll() is None:
                    self.send_message('log', f"No activity for {max_idle}s - terminating")
                    self._stop_process(process)
                    break

            stderr_text = ''.join(stderr_lines)
            returncode = process.poll()

            if self.should_stop.is_set():
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

    # Metadata Functions

    def save_video_metadata(self, video_data, folder):
        """Save video metadata to JSON file.

        Returns True if a new entry was actually added (deduplicated by id).
        """
        metadata_file = os.path.join(folder, "metadata.json")

        entry = self.create_metadata_entry(video_data)

        if not entry.get('id'):
            self.send_message('error', "Cannot save metadata: video ID is missing")
            return False

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
                    metadata = self._try_recover_corrupted_metadata(metadata_file)
            self._metadata_cache[cache_key] = metadata

        existing_ids = self._metadata_id_cache.setdefault(
            cache_key,
            {item.get('id') for item in metadata if isinstance(item, dict) and item.get('id')}
        )

        was_added = False
        if entry['id'] not in existing_ids:
            metadata.append(entry)
            existing_ids.add(entry['id'])
            self.current_downloads += 1
            was_added = True
        self._metadata_cache[cache_key] = metadata

        return was_added

    def _flush_metadata_cache(self, folder):
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
        channel_name = video_data.get('channel') or video_data.get('uploader') or ''

        entry = {
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

        if save_timestamp:
            entry['original_timestamp'] = self._sanitize_value(save_timestamp, str)

        return entry

    def update_metadata(self, folder):
        metadata_file = os.path.join(folder, "metadata.json")

        if not os.path.exists(metadata_file):
            self.send_message('error', f"No previous downloads found. Directory: {folder}")
            return

        videos_to_update = self.find_outdated_videos(metadata_file)

        if not videos_to_update:
            self.send_message('status', "All metadata is up to date (updated within 7 days)")
            return

        self.send_message('status', f"{len(videos_to_update)} videos need metadata updates")

        def parse_upload_date(item):
            try:
                upload_date = item.get('upload_date', '')
                if upload_date and len(upload_date) == 8:
                    return datetime.strptime(upload_date, '%Y%m%d')
            except:
                pass
            return datetime.min

        videos_to_update.sort(key=parse_upload_date, reverse=True)

        self.update_videos_metadata(videos_to_update, folder)

    def find_outdated_videos(self, metadata_file):
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
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=None)

            age = datetime.now() - timestamp
            return age > timedelta(days=days)

        except (ValueError, TypeError, AttributeError):
            return True

    def update_videos_metadata(self, videos, folder):
        if not self.is_yt_dlp_available():
            self.send_message('error', "yt-dlp is not installed or not found in PATH. Please install yt-dlp first.")
            return 0

        updated = 0
        total = len(videos)

        list_ = videos[0:total]
        updated += self.update_batch(list_, folder)

        if updated > 0:
            self.send_message('status', f"✓ Updated metadata for {updated}/{total} videos")
        else:
            self.send_message('status', "No metadata was updated")

    def update_batch(self, batch, folder):
        """Update a batch of videos.

        Uses --dump-json WITHOUT --skip-download (which would suppress output
        in modern yt-dlp) and without --print-json. --dump-json implies
        simulation by itself, so no files are actually downloaded.
        """
        urls = [f"https://www.youtube.com/watch?v={v['id']}" for v in batch]

        command = [
            self.get_yt_dlp_path(),
            "--dump-json",
            "--no-warnings",
            "--extractor-args", "youtube:player-client=mweb,tv,web_safari",
            "--remote-components", "ejs:github"
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
            except json.JSONDecodeError:
                continue

            if video_data.get('_type') == 'playlist':
                continue

            video_id = video_data.get('id')
            original = next((v for v in batch if v['id'] == video_id), None)

            if original:
                entry = self.create_metadata_entry(
                    video_data,
                    save_timestamp=original['timestamp']
                )

                self.update_metadata_entry(folder, entry)
                updated += 1

                progress = (updated / len(batch)) * 100
                self.send_message('progress', progress)

                days_old = self.calculate_days_since(original['timestamp'])
                message = self.format_update_message(entry, days_old)
                self.send_message('log', message)

        return updated

    def update_metadata_entry(self, folder, entry):
        metadata_file = os.path.join(folder, "metadata.json")

        try:
            metadata = self.load_json(metadata_file)

            for i, item in enumerate(metadata):
                if item.get('id') == entry['id']:
                    if 'original_timestamp' not in item and 'timestamp' in item:
                        entry['original_timestamp'] = item['timestamp']
                    elif 'original_timestamp' in item:
                        entry['original_timestamp'] = item['original_timestamp']

                    metadata[i] = entry
                    break

            self.save_json(metadata_file, metadata)

        except Exception as e:
            self.send_message('error', f"Could not update metadata: {str(e)}")

    # Consistency Check Functions

    def check_channel(self, folder, channel_name):
        issues_found = 0
        issues_fixed = 0

        self.send_message('log', f"\n=== Checking {channel_name} ===")

        metadata_ids, metadata = self.load_metadata_ids(folder)
        archive_ids, duplicate_archive_count = self.load_archive_ids(folder)
        vtx_ids = self.load_vtx_ids(folder)

        if duplicate_archive_count:
            issues_found += duplicate_archive_count
            self.send_message('log', f"  Issue: Found {duplicate_archive_count} duplicate archive entries")
            removed_duplicates = self.remove_duplicate_archive_entries(folder)
            if removed_duplicates:
                issues_fixed += removed_duplicates
                archive_ids, _ = self.load_archive_ids(folder)

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

        missing_archive = []
        for entry in metadata:
            video_id = entry.get('id')
            if not video_id:
                continue

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

        orphaned_metadata = []
        for entry in metadata:
            video_id = entry.get('id')
            if not video_id:
                continue

            if video_id not in vtx_ids and video_id not in archive_ids:
                orphaned_metadata.append(video_id)
                title = entry.get('title', video_id)
                self.send_message('log', f"  Issue: Metadata without archive or VTX: {title}")

        if orphaned_metadata:
            issues_found += len(orphaned_metadata)
            if self.remove_metadata_entries(folder, metadata, orphaned_metadata):
                issues_fixed += len(orphaned_metadata)

        missing_vtx = metadata_ids & archive_ids - vtx_ids
        if missing_vtx:
            self.send_message('log',
                f"  Info: {len(missing_vtx)} videos have metadata/archive but no VTX")

        if issues_found == 0:
            self.send_message('log', "  ✓ No issues found")

        return issues_found, issues_fixed

    def load_metadata_ids(self, folder):
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
        vtx_folder = os.path.join(folder, "vtx_files")

        if not os.path.exists(vtx_folder):
            return set()

        ids = set()
        for filename in os.listdir(vtx_folder):
            if filename.endswith('.vtx'):
                match = re.search(r'\[([^\]]+)\]\.vtx', filename)
                if match:
                    ids.add(match.group(1))

        return ids

    def remove_metadata_entries(self, folder, metadata, video_ids):
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
        archive_file = os.path.join(folder, "archive.txt")

        if not os.path.exists(archive_file):
            return 0

        try:
            with open(archive_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

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
        archive_file = os.path.join(folder, "archive.txt")

        try:
            if not os.path.exists(archive_file):
                with open(archive_file, 'w', encoding='utf-8') as f:
                    pass

            existing_ids = set()
            if os.path.exists(archive_file):
                with open(archive_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parsed_id = self._parse_archive_video_id(line)
                        if parsed_id:
                            existing_ids.add(parsed_id)

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
        stripped = line.strip()
        if not stripped:
            return None

        parts = stripped.split()
        if len(parts) >= 2 and parts[0].lower() == 'youtube':
            return parts[1]

        return None

    # Helper Functions

    def get_input_folder(self):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(script_dir, "data", "input")

    def get_channel_folder(self, channel_id):
        return os.path.join(self.get_input_folder(), channel_id)

    def get_all_channels(self, input_folder):
        return [d for d in os.listdir(input_folder)
                if os.path.isdir(os.path.join(input_folder, d))]

    def extract_channel_id(self, url):
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

        playlist_match = re.search(r'[&?]list=([^&]+)', url)
        if playlist_match:
            return playlist_match.group(1)

        return "youtube_subtitles"

    def load_json(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                data = self._deduplicate_entries(data)

            return data
        except json.JSONDecodeError as e:
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
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            valid_entries = []
            import re

            cleaned = re.sub(r'^\s*,\s*$', '', content, flags=re.MULTILINE)
            cleaned = re.sub(r'\},\s*"id"', r'},\n    {\n        "id"', cleaned)
            cleaned = re.sub(r'"([a-z_]+):\s*"', r'"\1": "', cleaned)
            cleaned = re.sub(r':\s*,', r': "",', cleaned)
            cleaned = re.sub(r':\s*"(https?[^"]*)\s*,', r': "",', cleaned)

            try:
                data = json.loads(cleaned)
                if isinstance(data, list):
                    data = self._deduplicate_entries(data)
                    if data:
                        self.send_message('status', f"Successfully recovered {len(data)} entries after cleaning corruption patterns")
                        return data
            except json.JSONDecodeError:
                pass

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

            pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(pattern, content)

            for match in matches:
                try:
                    fixed_match = re.sub(r':\s*"([^"]*?)\s*\n', r': "\1",\n', match)
                    fixed_match = re.sub(r':\s*,', r': "",', fixed_match)
                    fixed_match = re.sub(r'"([a-z_]+):\s*"', r'"\1": "', fixed_match)

                    entry = json.loads(fixed_match)
                    if isinstance(entry, dict) and 'id' in entry and entry['id']:
                        valid_entries.append(entry)
                except (json.JSONDecodeError, ValueError):
                    pass

            if valid_entries:
                valid_entries = self._deduplicate_entries(valid_entries)
                self.send_message('status', f"Recovered {len(valid_entries)} valid entries from corrupted metadata")
                return valid_entries

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
        self._validate_json_data(data)

        if os.path.exists(filepath):
            try:
                os.replace(filepath, filepath + '.backup')
            except:
                pass

        temp_filepath = filepath + '.tmp'
        try:
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            with open(temp_filepath, 'r', encoding='utf-8') as f:
                json.load(f)

            os.replace(temp_filepath, filepath)
        except Exception as e:
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            raise e

    def _is_truncated_value(self, val_str):
        return val_str.endswith((',', ':', '"'))

    def _has_field_corruption(self, val_str):
        if val_str.count(':') > 2:
            return True
        if val_str.count('"') > 2 and ':' in val_str:
            return True
        return False

    def _extract_duplicated_content(self, val_str):
        if ': "' not in val_str:
            return None

        parts = val_str.split(': "', 1)
        if len(parts) != 2:
            return None

        prefix, rest = parts[0].strip(), parts[1]
        if prefix and len(prefix) > 10:
            prefix_words = prefix.split()
            if prefix_words:
                first_words = ' '.join(prefix_words[:min(3, len(prefix_words))])
                if first_words and rest.startswith(first_words):
                    return rest
        return None

    def _validate_url(self, val_str):
        if not val_str.startswith('http'):
            return val_str
        if self._is_truncated_value(val_str):
            return ''
        if len(val_str) < 25:
            return ''
        if 'youtube.com/watch' not in val_str and 'youtube.com/channel' not in val_str:
            return ''
        return val_str

    def _validate_date(self, val_str):
        if not (val_str.isdigit() or '-' in val_str):
            return val_str
        if val_str.replace('-', '').isdigit():
            date_digits = val_str.replace('-', '')
            if len(date_digits) < 8 or self._is_truncated_value(val_str):
                return ''
            return val_str
        return val_str

    def _validate_channel_id(self, val_str):
        if not val_str.startswith('UC'):
            return val_str
        if len(val_str) < 24 or self._is_truncated_value(val_str):
            return ''
        return val_str

    def _validate_video_id(self, val_str):
        if len(val_str) != 11:
            return val_str
        if val_str.replace('-', '').replace('_', '').isalnum():
            return val_str
        return ''

    def _sanitize_value(self, value, expected_type):
        if value is None:
            return {str: '', int: 0, bool: False}.get(expected_type, value)

        try:
            if expected_type == str:
                val_str = str(value).strip()

                if val_str.startswith((':', ',', '"', '|')):
                    return ''

                if self._has_field_corruption(val_str):
                    return ''

                cleaned = self._extract_duplicated_content(val_str)
                if cleaned:
                    return cleaned

                if val_str.startswith('http'):
                    return self._validate_url(val_str)
                if val_str and (val_str.isdigit() or '-' in val_str):
                    return self._validate_date(val_str)
                if val_str.startswith('UC'):
                    return self._validate_channel_id(val_str)
                if len(val_str) == 11:
                    return self._validate_video_id(val_str)

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
        if not isinstance(data, list):
            return

        cleaned_data = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            if not item.get('id'):
                self.send_message('warning', f"Entry {i} has no ID, skipping")
                continue

            if self._is_entry_severely_corrupted(item):
                self.send_message('warning', f"Entry {i} (ID: {item.get('id')}) is severely corrupted, skipping")
                continue

            cleaned_item = self._clean_entry_fields(item)
            if cleaned_item.get('id'):
                cleaned_data.append(cleaned_item)
            else:
                self.send_message('warning', f"Entry {i} lost required ID during cleaning, skipping")

        data.clear()
        data.extend(cleaned_data)

    def _clean_entry_fields(self, item):
        cleaned_item = {}
        seen_keys = set()

        for key, value in item.items():
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cleaned_item[key] = self._clean_field_value(key, value)

        for field, default_val in self.CRITICAL_FIELDS.items():
            if field not in cleaned_item:
                cleaned_item[field] = default_val

        return cleaned_item

    def _clean_field_value(self, key, value):
        if value is None:
            return ''

        if isinstance(value, str):
            if value.endswith((',', ':', '"')):
                return ''
            if value.startswith('http') and len(value) < 20:
                return ''
            if value.startswith('UC') and len(value) < 24:
                return ''
            return value

        if not isinstance(value, (int, float, bool)):
            try:
                str_val = str(value)
                return '' if str_val.endswith((',', ':')) else str_val
            except:
                return ''

        return value

    def _is_entry_severely_corrupted(self, item):
        if not isinstance(item, dict):
            return True

        id_value = item.get('id', '')
        if isinstance(id_value, str):
            if id_value.startswith((':', ',', '"')):
                return True
            if id_value.count(':') > 1 or len(id_value) > 200 or '{' in id_value:
                return True
            if id_value.endswith((':', 'GROU:', 'PLAYGRO:', 'PLAYGROUND')):
                if not (len(id_value) == 11 and id_value.replace('-', '').replace('_', '').isalnum()):
                    return True

        suspicious_keys = self._count_suspicious_keys(item)
        if suspicious_keys >= 2:
            return True

        title_value = item.get('title', '')
        if isinstance(title_value, str):
            if '": "' in title_value or title_value.startswith('"') or title_value.endswith('":'):
                return True
            if self._has_duplicated_content(title_value):
                return True

        if self._has_merged_field_data(item):
            return True

        return False

    def _count_suspicious_keys(self, item):
        truncated_fields = ('uplo', 'vie', 'tim', 'cha', 'sub', 'com', 'lik', 'was')
        suspicious_count = 0

        for key in item.keys():
            if not isinstance(key, str):
                return float('inf')
            if '"' in key or "'" in key or key.startswith(':'):
                return float('inf')
            if key in truncated_fields:
                suspicious_count += 1

        return suspicious_count

    def _has_duplicated_content(self, val_str):
        if ': "' not in val_str:
            return False

        parts = val_str.split(': "', 1)
        if len(parts) != 2:
            return False

        prefix, rest = parts[0], parts[1]
        if not prefix or len(prefix) <= 10:
            return False

        prefix_words = ' '.join(prefix.split()[:3])
        return bool(prefix_words and prefix_words in rest and len(prefix_words) > 10)

    def _has_merged_field_data(self, item):
        for key, value in item.items():
            if isinstance(value, str):
                if ': "' in value and len(value) > 50:
                    return True
        return False

    def _deduplicate_entries(self, data):
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
                        continue
                    seen_ids.add(video_id)

            deduplicated.append(item)

        if duplicates_removed > 0:
            self.send_message('status', f"Removed {duplicates_removed} duplicate entries from metadata")

        return deduplicated

    # Formatting Functions

    def format_download_message(self, video_data):
        title = video_data.get('title', 'Unknown')
        upload_date = video_data.get('upload_date', '')
        view_count = video_data.get('view_count', 0)
        category = video_data.get('categories', [''])[0] if video_data.get('categories') else ''
        channel = video_data.get('channel') or video_data.get('uploader', '')

        status = []
        if video_data.get('is_live'):
            status.append("LIVE")
        elif video_data.get('was_live'):
            status.append("Was Live")

        status_str = f" [{', '.join(status)}]" if status else ""
        date_str = self.format_date(upload_date)

        message = f"Downloaded: {title}{status_str} ({date_str})"

        if view_count:
            message += f" | Views: {self.format_number(view_count)}"
        if category:
            message += f" | Category: {category}"
        if channel:
            message += f" | Channel: {channel}"

        return message

    def format_update_message(self, video_data, days_old):
        title = video_data.get('title', 'Unknown')
        view_count = video_data.get('view_count', 0)
        category = video_data.get('category', '')
        channel = video_data.get('channel_name', '')

        status = []
        if video_data.get('is_live'):
            status.append("LIVE")
        elif video_data.get('was_live'):
            status.append("Was Live")

        status_str = f" [{', '.join(status)}]" if status else ""

        message = f"Updated: {title}{status_str} ({days_old} days old)"

        if view_count:
            message += f" | Views: {self.format_number(view_count)}"
        if category:
            message += f" | Category: {category}"
        if channel:
            message += f" | Channel: {channel}"

        return message

    def format_date(self, date_str):
        if not date_str or len(date_str) != 8:
            return date_str

        try:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except:
            return date_str

    def format_number(self, number):
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
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=None)

            days = (datetime.now() - timestamp).days
            return str(days)
        except (ValueError, TypeError, AttributeError):
            return "?"

    # Message Queue Functions

    def send_message(self, msg_type, content):
        self.messages.put((msg_type, content))

    def check_messages(self):
        try:
            while True:
                msg_type, content = self.messages.get_nowait()
                self.handle_message(msg_type, content)
        except queue.Empty:
            pass

        self.after(100, self.check_messages)

    def handle_message(self, msg_type, content):
        if msg_type == 'status':
            self.status_text.set(content)
        elif msg_type == 'progress':
            self.progress_bar['value'] = content
        elif msg_type == 'log':
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
        elif msg_type == 'done':
            self.enable_buttons()


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    downloader = SubtitleDownloader(root)
    downloader.protocol("WM_DELETE_WINDOW", root.destroy)

    root.mainloop()

    #caffeinate -i python3 core/atlas_downloader.py

#cd /Users/c/bgutil-ytdlp-pot-provider/server
#deno run --allow-net --allow-read --allow-env --allow-ffi=. src/main.ts
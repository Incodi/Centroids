"""
atlas_downloader.py  (lives in <project>/core/)

Downloads YouTube auto-subtitles with yt-dlp into:
    <project>/data/input/<channel>/vtt_files   (staging, converted to VTX afterwards)
    <project>/data/input/<channel>/vtx_files
    <project>/data/input/<channel>/metadata.json
    <project>/data/input/<channel>/archive.txt

How it works
------------
* One yt-dlp run per channel. It uses --lazy-playlist and, by default,
  --break-on-existing to stop at the first archived video instead of walking older uploads.
  --max-downloads also caps the number of new videos processed. There is NO separate pre-scan.
* --print after_video:<json> --no-simulate prints AFTER the subtitle file is written
  (the old --print-json printed before, so failures looked like successes).
* After the run we verify every video yt-dlp archived: if its .vtt is not on disk it is
  removed from archive.txt again, so a failed/rate-limited fetch is retried next time.
* stdout is read to EOF, stderr is drained on its own thread, files go to -P <vtt_dir>.
"""

from datetime import datetime, timedelta
import json
import os
import queue
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Make `helpers.vtt_helper` importable when this file is run from core/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from helpers.vtt_helper import VTTSegment
except ImportError:
    from vtt_helper import VTTSegment


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SUB_LANGS = "en"                       # --sub-langs
SUBTITLE_SLEEP = 3                     # seconds before each subtitle request
SLEEP_REQUESTS = 1                     # seconds between extraction requests
COOKIES_FROM_BROWSER = "firefox"       # yt-dlp browser-cookie syntax; use None to disable
CHANNEL_PAUSE = (8, 20)                # random pause between channels in batch mode (seconds)
REQUIRE_IMPERSONATION = True           # refuse to run without a curl_cffi impersonation target
RATE_LIMIT_COOLDOWN_MIN = 60           # after a 429, refuse to start again for this long (0 = off)
MAX_FAILURES = 3                       # after this many failures a video stays archived (given up) until "Retry" is ticked
IDLE_TIMEOUT = 600                     # kill yt-dlp if it prints nothing for this long (seconds)
STOP_AT_FIRST_ARCHIVED = True          # Stop at the first archived video (new uploads only; no older backfill)
BGUTIL_URL = "http://127.0.0.1:4416"   # bgutil PO-token HTTP provider
PLAYER_CLIENTS = "web_safari" #"default,mweb"        # what the old working downloader used; None = let yt-dlp choose
NETWORK_RESUMES = 3                    # re-run the download this many times after a network drop
DELETE_VTT_AFTER_CONVERT = True        # delete a VTT once its VTX + TXT exist
REMOVE_EMPTY_VTT_DIR = False           # keep vtt_files/ around so you can see it

USE_CONFIG = object()                  # sentinel: base_args() uses the configured player clients
DIAG_VIDEO = "https://www.youtube.com/watch?v=jNQXAC9IVRw"   # public test video used by Diagnose

VIDEO_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')
META_FIELDS = (
    "id,title,webpage_url,upload_date,duration,view_count,like_count,comment_count,"
    "was_live,is_live,channel,uploader,channel_id,channel_url,channel_follower_count,categories"
)
PRINT_TEMPLATE = "%(.{" + META_FIELDS + "})j"   # compact JSON, not the multi-MB full info dict

FAIL_MARKERS = (
    "unable to download video subtitles", "http error", "error:", "sign in",
    "not a bot", "timed out", "unable to download", "po token",
)
RATE_LIMIT_MARKERS = ("http error 429", "too many requests")
NETWORK_ERROR_MARKERS = (
    "network is unreachable", "no route to host", "temporary failure in name resolution",
    "name or service not known", "failed to resolve", "could not resolve host",
    "connection timed out", "timed out", "connection reset by peer",
    "connection refused", "socket timeout", "unable to download webpage", "proxyerror",
)


class SubtitleDownloader(tk.Toplevel):

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.title("Subtitle Downloader")
        self.minsize(820, 600)
        self.center_window(1000, 750)

        self.max_downloads = 100
        self.batch_download_number = 100
        self.batch_filter_handles = []

        # Threading / process state
        self.worker_thread = None
        self.should_stop = threading.Event()
        self.messages = queue.Queue()
        self._current_proc = None
        self._proc_lock = threading.Lock()
        self.rate_limited = False
        self.opts = {}

        # Per-channel metadata working set (worker thread only)
        self._meta_entries = []
        self._meta_ids = set()

        # Tool discovery (filled in by preflight)
        self.env = self._build_env()
        self._yt_dlp_path = None
        self._deno_path = None
        self._use_bgutil = False
        self._clients = PLAYER_CLIENTS

        self.style = ttk.Style(self)
        self.create_widgets()
        self.after(100, self.check_messages)

    # ------------------------------------------------------------------ UI
    def center_window(self, width, height):
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 3, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def create_widgets(self):
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
        self.progress_bar = ttk.Progressbar(
            container, style="red.Horizontal.TProgressbar", mode='determinate')
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
        opts_frame = ttk.Frame(parent)
        opts_frame.grid(column=0, row=2, columnspan=3, sticky=tk.W, pady=(5, 10))

        self.metadata_only = tk.BooleanVar()
        ttk.Checkbutton(
            opts_frame,
            text="Update metadata only (doesn't download new subtitles)",
            variable=self.metadata_only
        ).grid(column=0, row=0, sticky=tk.W)

        self.retry_missing = tk.BooleanVar()
        ttk.Checkbutton(
            opts_frame,
            text="Retry videos that have no transcript on disk (ignore archive/metadata)",
            variable=self.retry_missing
        ).grid(column=1, row=0, sticky=tk.W, padx=(16, 0))

        self.stop_at_archived = tk.BooleanVar(value=STOP_AT_FIRST_ARCHIVED)
        ttk.Checkbutton(
            opts_frame,
            text="New uploads only (recommended): stop at first archived video; uncheck to backfill older videos",
            variable=self.stop_at_archived
        ).grid(column=0, row=1, columnspan=2, sticky=tk.W, pady=(4, 0))

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
            batch_frame, text="Load Handle List (.txt)", command=self.load_batch_handle_list)
        self.batch_filter_button.grid(column=2, row=0, sticky=tk.W, padx=(16, 0))

    def create_buttons(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.grid(column=0, row=4, columnspan=3, sticky=tk.W, pady=(10, 0))

        self.download_button = ttk.Button(
            button_frame, text="Download Subtitles", command=self.start_download)
        self.download_button.grid(column=0, row=0, sticky=tk.W)

        self.consistency_button = ttk.Button(
            button_frame, text="Check Consistency", command=self.start_consistency_check)
        self.consistency_button.grid(column=1, row=0, sticky=tk.W, padx=(20, 0))

        self.stop_button = ttk.Button(
            button_frame, text="Stop", command=self.request_stop, state=tk.DISABLED)
        self.stop_button.grid(column=2, row=0, sticky=tk.W, padx=(20, 0))

        self.diagnose_button = ttk.Button(
            button_frame, text="Diagnose (1 video)", command=self.start_diagnose)
        self.diagnose_button.grid(column=3, row=0, sticky=tk.W, padx=(20, 0))

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

    # ------------------------------------------------------- Main control
    def start_download(self):
        if self.is_working():
            messagebox.showwarning("Warning", "Operation already in progress!", parent=self)
            return

        # Snapshot Tk variables on the UI thread; the worker never touches Tk widgets.
        self.opts = {
            'url': self.url_input.get().strip(),
            'batch': self.batch_mode.get(),
            'metadata_only': self.metadata_only.get(),
            'reverse': self.reverse_batch_order.get(),
            'retry': self.retry_missing.get(),
            'stop_first': self.stop_at_archived.get(),
        }
        self.should_stop.clear()
        self.rate_limited = False
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

    def request_stop(self):
        self.should_stop.set()
        with self._proc_lock:
            proc = self._current_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        self.status_text.set("Stopping...")

    def is_working(self):
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def prepare_for_download(self):
        self.disable_buttons()
        self.progress_bar['value'] = 0
        self.output.delete(1.0, tk.END)

        if self.opts['batch']:
            text = "Batch Downloading..."
        elif self.opts['metadata_only']:
            text = "Updating Metadata..."
        else:
            text = "Downloading..."
        self.download_button.config(text=text)

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
        self.diagnose_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def enable_buttons(self):
        self.download_button.config(state=tk.NORMAL, text="Download Subtitles")
        self.consistency_button.config(state=tk.NORMAL, text="Check Consistency")
        self.batch_filter_button.config(state=tk.NORMAL)
        self.diagnose_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    # ------------------------------------------------------ Worker threads
    def download_worker(self):
        try:
            if not self.is_yt_dlp_available():
                self.send_message('error', "yt-dlp is not installed or not found. Please install yt-dlp first.")
                return

            remaining = self._cooldown_remaining_minutes()
            if remaining:
                self.send_message(
                    'error',
                    f"YouTube rate-limited this IP recently. Waiting ~{remaining} more minutes before "
                    f"starting avoids extending the block. (Override: delete data/.rate_limit_until "
                    f"or set RATE_LIMIT_COOLDOWN_MIN = 0)")
                return

            if not self.preflight():
                return

            if self.opts['batch']:
                self.process_batch_download()
            else:
                url = self.opts['url']
                if not url:
                    self.send_message('error', "Please enter a YouTube channel or playlist URL!")
                    return

                folder = self.get_channel_folder(self.extract_channel_id(url))
                if self.opts['metadata_only']:
                    self.update_metadata(folder)
                else:
                    self.download_from_channel(folder, url, self.max_downloads, show_progress=True)

        except Exception as e:
            self.send_message('error', f"Unexpected error: {e}")
            self.send_message('log', traceback.format_exc())
        finally:
            self.send_message('done', None)

    # ---------------------------------------------------------- Diagnose
    def start_diagnose(self):
        if self.is_working():
            messagebox.showwarning("Warning", "Operation already in progress!", parent=self)
            return
        self.opts = {'url': self.url_input.get().strip()}
        self.should_stop.clear()
        self.disable_buttons()
        self.output.delete(1.0, tk.END)
        self.progress_bar['value'] = 0
        self.status_text.set("Running subtitle diagnostics on one video...")
        self.worker_thread = threading.Thread(target=self.diagnose_worker, daemon=True)
        self.worker_thread.start()

    def diagnose_worker(self):
        """Fetch ONE video's subtitles with several player-client settings and report what happens."""
        try:
            if not self.is_yt_dlp_available():
                self.send_message('error', "yt-dlp is not installed or not found.")
                return
            remaining = self._cooldown_remaining_minutes()
            if remaining:
                self.send_message(
                    'warning',
                    f"YouTube rate-limited this IP recently. Wait ~{remaining} more minutes before running "
                    "diagnostics to avoid sending another request.")
                return
            if not self.preflight():
                return

            url = self.opts.get('url', '')
            if not re.search(r'youtube\.com/watch\?v=|youtu\.be/', url):
                url = DIAG_VIDEO
                self.send_message('log', "No video URL in the box; using a public test video.")
            self.send_message('log', f"Testing one video: {url}")

            variants = [
                ("current settings", self._clients),
                ("tv client only", "tv"),
                ("android_vr client only", "android_vr"),
                ("web_safari client only", "web_safari"),
            ]
            first_ok = None
            current_ok = False
            diagnostic_429 = False
            for i, (label, clients) in enumerate(variants):
                if self.should_stop.is_set():
                    break
                self.send_message('status', f"Diagnosing: {label}...")
                ok, lines = self._diagnose_once(url, clients)
                self.send_message('log', f"\n--- {label} (clients: {clients or 'yt-dlp default'}) ---")
                for line in lines:
                    self.send_message('log', "  " + line)
                self.send_message('log', "RESULT: subtitle file saved ✓" if ok else "RESULT: no subtitle file ✗")
                if ok and i == 0:
                    current_ok = True
                if ok and first_ok is None:
                    first_ok = (label, clients)
                if self._has_marker("\n".join(lines), RATE_LIMIT_MARKERS):
                    diagnostic_429 = True
                    self._set_cooldown()
                self.send_message('progress', ((i + 1) / len(variants)) * 100)
                if ok or diagnostic_429:
                    break
                self._sleep_interruptible(3)

            self.send_message('log', "\n=== Summary ===")
            if current_ok:
                self.send_message(
                    'log', "Current settings downloaded subtitles for this test video once. That does not rule out "
                           "HTTP 429 for another video or after earlier requests. A 429 alone cannot distinguish "
                           "request throttling from a client/token rejection. Diagnose the exact failing video "
                           "after the cooldown; diagnostics stop at the first client that works.")
            elif first_ok:
                self._clients = first_ok[1]
                self.send_message(
                    'log', f"'{first_ok[0]}' works and the current settings do not. Using clients="
                           f"'{first_ok[1]}' for the rest of this session. To keep it, set "
                           f"PLAYER_CLIENTS = \"{first_ok[1]}\" at the top of the file.")
            elif diagnostic_429:
                self.send_message(
                    'log', "HTTP 429 encountered; stopped probing other clients to avoid adding more requests. "
                           "A 429 alone cannot distinguish request throttling from a client/token rejection. "
                           "Wait for the cooldown before retrying.")
            else:
                self.send_message(
                    'log', "No setting produced a subtitle file. Look at the pot/bgutil lines above. Check:\n"
                           f"  curl {BGUTIL_URL}/ping     (server answers and shows its version)\n"
                           "  pipx runpip yt-dlp list | grep -i -E 'bgutil|curl'   (plugin installed in yt-dlp's venv; "
                           "its version should match the server's)\n"
                           "  then restart the bgutil server and run Diagnose again.")
            self.send_message('status', "Diagnostics finished (see log)")
        except Exception as e:
            self.send_message('error', f"Diagnose error: {e}")
            self.send_message('log', traceback.format_exc())
        finally:
            self.send_message('done', None)

    def _diagnose_once(self, url, clients):
        tmp = tempfile.mkdtemp(prefix="atlas_diag_")
        command = self.base_args(clients=clients) + [
            "-v", "--skip-download",
            "--write-subs", "--write-auto-subs",
            "--sub-langs", SUB_LANGS, "--sub-format", "vtt",
            "-P", tmp, "-o", "%(id)s.%(ext)s", url,
        ]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=240, env=self.env)
            text = proc.stdout + "\n" + proc.stderr
        except subprocess.TimeoutExpired:
            text = "yt-dlp timed out after 240s"
        except Exception as e:
            text = f"could not run yt-dlp: {e}"

        try:
            ok = any(f.endswith(".vtt") for f in os.listdir(tmp))
        except OSError:
            ok = False
        shutil.rmtree(tmp, ignore_errors=True)

        keys = re.compile(r'pot|po token|bgutil|impersonat|subtitle|429|error|warning|writing', re.I)
        lines = [l.strip()[:220] for l in text.splitlines()
                 if keys.search(l) and 'Command-line' not in l and l.strip()]
        return ok, lines[:16]

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
            total_issues = total_fixed = 0

            for i, channel in enumerate(channels):
                if self.should_stop.is_set():
                    break
                issues, fixed = self.check_channel(os.path.join(input_folder, channel), channel)
                total_issues += issues
                total_fixed += fixed
                self.send_message('progress', ((i + 1) / len(channels)) * 100)

            self.send_message('status', f"✓ Found {total_issues} issues, fixed {total_fixed}")

        except Exception as e:
            self.send_message('error', f"Consistency check error: {e}")
            self.send_message('log', traceback.format_exc())
        finally:
            self.send_message('done', None)

    # ------------------------------------------------- Tool discovery / env
    def _build_env(self):
        """GUI launches often miss shell PATH additions (deno, pipx, homebrew)."""
        env = os.environ.copy()
        extra = [
            os.path.expanduser("~/.deno/bin"),
            os.path.expanduser("~/.local/bin"),
            "/opt/homebrew/bin",
            "/usr/local/bin",
        ]
        env["PATH"] = os.pathsep.join([p for p in extra if os.path.isdir(p)] + [env.get("PATH", "")])
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def get_yt_dlp_path(self):
        if self._yt_dlp_path:
            return self._yt_dlp_path

        # pipx installs first: they have the bgutil PO-token plugin injected.
        candidates = [
            os.path.expanduser("~/Library/Application Support/pipx/venvs/yt-dlp/bin/yt-dlp"),
            os.path.expanduser("~/.local/pipx/venvs/yt-dlp/bin/yt-dlp"),
            os.path.expanduser("~/.local/bin/yt-dlp"),
        ]
        for path in candidates:
            if os.path.exists(path):
                self._yt_dlp_path = path
                return path

        found = shutil.which("yt-dlp", path=self.env["PATH"])
        if found:
            self._yt_dlp_path = found
            return found

        framework = "/Library/Frameworks/Python.framework/Versions/3.13/bin/yt-dlp"
        if os.path.exists(framework):
            self._yt_dlp_path = framework
            return framework

        return "yt-dlp"

    def is_yt_dlp_available(self):
        path = self.get_yt_dlp_path()
        if path == "yt-dlp":
            return shutil.which("yt-dlp", path=self.env["PATH"]) is not None
        return os.path.exists(path)

    def _bgutil_reachable(self):
        m = re.match(r'https?://([^:/]+):?(\d+)?', BGUTIL_URL)
        if not m:
            return False
        host, port = m.group(1), int(m.group(2) or 80)
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            return False

    def _impersonation_available(self):
        """True if yt-dlp has at least one usable browser-impersonation target (curl_cffi)."""
        try:
            proc = subprocess.run(
                [self.get_yt_dlp_path(), "--list-impersonate-targets"],
                capture_output=True, text=True, timeout=30, env=self.env)
        except Exception:
            return False
        for line in (proc.stdout + "\n" + proc.stderr).splitlines():
            low = line.lower()
            if "curl_cffi" in low and "unavailable" not in low:
                return True
        return False

    def preflight(self):
        """Log what we are running with. Returns False if a known-fatal problem is found."""
        try:
            version = subprocess.run(
                [self.get_yt_dlp_path(), "--version"], capture_output=True, text=True,
                timeout=20, env=self.env).stdout.strip()
        except Exception:
            version = "unknown"
        self.send_message('log', f"yt-dlp {version}  ({self.get_yt_dlp_path()})")

        self._deno_path = shutil.which("deno", path=self.env["PATH"])
        if self._deno_path:
            self.send_message('log', f"JS runtime: deno ({self._deno_path})")
        else:
            self.send_message('warning', "deno not found on PATH: YouTube JS challenges will likely fail")

        self._use_bgutil = self._bgutil_reachable()
        if self._use_bgutil:
            self.send_message('log', f"PO-token provider reachable at {BGUTIL_URL}")
        else:
            self.send_message(
                'warning',
                f"No PO-token server at {BGUTIL_URL}. If subtitles fail with HTTP 429 / empty files, "
                f"start the bgutil provider.")

        if self._impersonation_available():
            self.send_message('log', "Browser impersonation: available (curl_cffi)")
        else:
            msg = ("No impersonation target available. YouTube subtitle requests will get HTTP 429 without it. "
                   "Fix: pipx inject yt-dlp \"yt-dlp[curl-cffi]\"   then check: yt-dlp --list-impersonate-targets")
            if REQUIRE_IMPERSONATION:
                self.send_message('error', msg + "  (Override: REQUIRE_IMPERSONATION = False)")
                return False
            self.send_message('warning', msg)
        return True

    def base_args(self, clients=USE_CONFIG):
        args = [
            self.get_yt_dlp_path(),
            "--color", "never",
            "--socket-timeout", "30",
            "--retries", "2",
            "--extractor-retries", "1",
            "--sleep-requests", str(SLEEP_REQUESTS),
            "--remote-components", "ejs:github",
            # We never download media: skip the DASH/HLS manifest requests.
            "--extractor-args", "youtube:skip=dash,hls",
            "--extractor-args", "youtube:fetch_pot=always",
            "--ignore-no-formats-error",
        ]
        if COOKIES_FROM_BROWSER:
            args += ["--cookies-from-browser", COOKIES_FROM_BROWSER]
        if self._deno_path:
            args += ["--js-runtimes", f"deno:{self._deno_path}"]
        chosen = self._clients if clients is USE_CONFIG else clients
        if chosen:
            args += ["--extractor-args", f"youtube:player-client={chosen}"]
        if self._use_bgutil:
            args += ["--extractor-args", f"youtubepot-bgutilhttp:base_url={BGUTIL_URL}"]
        return args

    # Persistent cooldown so re-running right after a 429 doesn't extend the block.
    def _cooldown_path(self):
        return os.path.join(_ROOT, "data", ".rate_limit_until")

    def _cooldown_remaining_minutes(self):
        try:
            with open(self._cooldown_path(), 'r') as f:
                until = datetime.fromisoformat(f.read().strip())
        except (OSError, ValueError):
            return 0
        return max(0, int((until - datetime.now()).total_seconds() // 60) + 1) if until > datetime.now() else 0

    def _set_cooldown(self):
        if RATE_LIMIT_COOLDOWN_MIN <= 0:
            return
        try:
            os.makedirs(os.path.dirname(self._cooldown_path()), exist_ok=True)
            with open(self._cooldown_path(), 'w') as f:
                f.write((datetime.now() + timedelta(minutes=RATE_LIMIT_COOLDOWN_MIN)).isoformat())
        except OSError:
            pass

    def _sleep_interruptible(self, seconds):
        end = time.time() + seconds
        while time.time() < end and not self.should_stop.is_set():
            time.sleep(0.5)

    # ----------------------------------------------------- Process helpers
    def _spawn(self, command, kill_on_rate_limit=False):
        """Start a process; stderr is drained on a thread so pipes never deadlock."""
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", env=self.env)
        with self._proc_lock:
            self._current_proc = proc
        self._last_activity = time.time()

        err_lines = []

        def drain():
            try:
                for line in proc.stderr:
                    line = line.rstrip("\n")
                    self._last_activity = time.time()
                    err_lines.append(line)
                    # Stop immediately on HTTP 429: carrying on only digs the hole deeper.
                    if kill_on_rate_limit and self._has_marker(line, RATE_LIMIT_MARKERS):
                        try:
                            proc.terminate()
                        except Exception:
                            pass
            except Exception:
                pass

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        return proc, err_lines, thread

    def _finish(self, proc, thread):
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._stop_process(proc)
        thread.join(timeout=5)
        with self._proc_lock:
            if self._current_proc is proc:
                self._current_proc = None

    def _start_watchdog(self, proc, idle_timeout):
        """Kill yt-dlp if it produces no output at all for `idle_timeout` seconds."""
        done = threading.Event()

        def run():
            while not done.wait(10):
                if proc.poll() is not None:
                    return
                if time.time() - self._last_activity > idle_timeout:
                    self.send_message('log', f"  No output for {idle_timeout}s - terminating yt-dlp")
                    self._stop_process(proc)
                    return

        threading.Thread(target=run, daemon=True).start()
        return done

    def _stop_process(self, process):
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    # ------------------------------------------------------------- Network
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

    @staticmethod
    def _has_marker(text, markers):
        low = text.lower()
        return any(m in low for m in markers)

    # ------------------------------------------------------- Batch download
    def process_batch_download(self):
        input_folder = self.get_input_folder()
        if not os.path.exists(input_folder):
            self.send_message('error', f"Input directory not found: {input_folder}")
            return

        channels = self.get_all_channels(input_folder)
        channels.sort(key=str.lower)
        if not channels:
            self.send_message('error', "No channel directories found")
            return

        channels = self.apply_batch_filter(channels)
        if not channels:
            self.send_message('error', "No channels matched the loaded handle list")
            return

        reverse = self.opts['reverse']
        if reverse:
            channels.reverse()

        self.send_message('status', f"Processing {len(channels)} channels ({'Z→0' if reverse else '0→Z'})")

        for i, channel in enumerate(channels):
            if self.should_stop.is_set():
                break
            if self.rate_limited:
                self.send_message('warning', "Subtitle requests are being rejected (HTTP 429). Stopping the batch.")
                break

            self.process_single_channel_batch(
                os.path.join(input_folder, channel), channel, i + 1, len(channels))
            self.send_message('progress', ((i + 1) / len(channels)) * 100)
            if i + 1 < len(channels):
                self._sleep_interruptible(random.uniform(*CHANNEL_PAUSE))

        if not self.should_stop.is_set() and not self.rate_limited:
            self.send_message('status', f"✓ Batch complete. Processed {len(channels)} channels")

    def process_single_channel_batch(self, folder, channel_name, current, total):
        metadata_file = os.path.join(folder, "metadata.json")
        try:
            metadata = []
            if os.path.exists(metadata_file):
                try:
                    metadata = self.load_json(metadata_file)
                except Exception as e:
                    self.send_message('warning', f"Could not read metadata for {channel_name}: {e}")

            if not metadata:
                self.send_message('warning', f"No usable metadata for {channel_name}; seeding minimal metadata")
                metadata = [self._make_minimal_channel_metadata(channel_name)]
                try:
                    self.save_json(metadata_file, metadata)
                except Exception as e:
                    self.send_message('warning', f"Could not write seed metadata for {channel_name}: {e}")

            url = self.build_channel_url(channel_name, metadata)
            if not url:
                self.send_message('log', f"Skipping {channel_name}: Could not build URL")
                return

            self.send_message('log', f"[{current}/{total}] Processing: {channel_name}")
            self.download_from_channel(folder, url, self.batch_download_number, show_progress=False)

        except Exception as e:
            self.send_message('log', f"Error processing {channel_name}: {e}")

    # ----------------------------------------------------- Handle list
    def load_batch_handle_list(self):
        file_path = filedialog.askopenfilename(
            parent=self, title="Select channel handle list",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {e}", parent=self)
            return

        handles = self.parse_handles_from_text(text)
        if not handles:
            messagebox.showwarning(
                "No Handles Found", "No valid channel handles were found in the selected file.", parent=self)
            return

        created = 0
        normalized = []
        pattern = re.compile(
            r'^(?:https?://)?(?:www\.)?youtube\.com/@(?P<handle>[A-Za-z0-9_\-\.]+)/*$', re.IGNORECASE)

        for h in handles:
            m = pattern.match(h.strip())
            if not m:
                normalized.append(h)
                continue

            handle_name = m.group('handle')
            folder = os.path.join(self.get_input_folder(), handle_name)
            try:
                if not os.path.exists(folder):
                    os.makedirs(folder, exist_ok=True)
                    created += 1

                metadata_file = os.path.join(folder, 'metadata.json')
                existing = None
                if os.path.exists(metadata_file):
                    try:
                        existing = self.load_json(metadata_file)
                    except Exception:
                        existing = None
                if not existing:
                    self.save_json(metadata_file, [self._make_minimal_channel_metadata(handle_name)])
                normalized.append(handle_name)
            except Exception as e:
                self.send_message('log', f"Error preparing folder for {h}: {e}")
                normalized.append(h)

        self.batch_filter_handles = normalized
        self.batch_filter_button.config(text=f"Handle List: {len(normalized)}")
        if created:
            self.status_text.set(f"Loaded {len(normalized)} handles ({created} new channels created)")
        else:
            self.status_text.set(f"Loaded {len(normalized)} handles for batch filter")

    def parse_handles_from_text(self, text):
        """Parse handles from plain lines or '[x/y] Processing: HANDLE' lines"""
        handles, seen = [], set()
        for raw in text.splitlines():
            line = raw.strip()
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

        available = {name.casefold(): name for name in channels}
        selected = [available[h.casefold()] for h in self.batch_filter_handles if h.casefold() in available]

        missing = [h for h in self.batch_filter_handles if h.casefold() not in available]
        if missing:
            preview = ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")
            self.send_message('log', f"Handle list entries not found in input directory: {preview}")

        self.send_message('log', f"Batch filter active: selected {len(selected)} of {len(channels)} channels")
        return selected

    def build_channel_url(self, channel_name, metadata):
        if metadata:
            for entry in metadata:
                if isinstance(entry, dict):
                    url = (entry.get('channel_url') or '').strip()
                    if url:
                        return url
            for entry in metadata:
                if isinstance(entry, dict):
                    cid = (entry.get('channel_id') or '').strip()
                    if cid:
                        return f"https://www.youtube.com/channel/{cid}"
        if channel_name:
            return f"https://www.youtube.com/@{channel_name.lstrip('@')}"
        return None

    def _make_minimal_channel_metadata(self, channel_name):
        safe = channel_name.strip().lstrip('@')
        if safe.startswith('UC') and len(safe) >= 24:
            url, cid = f"https://www.youtube.com/channel/{safe}", safe
        else:
            url, cid = f"https://www.youtube.com/@{safe}", ''
        return {
            'id': safe, 'title': safe, 'url': url,
            'timestamp': datetime.now().isoformat(),
            'channel_name': safe, 'channel_url': url, 'channel_id': cid,
        }

    # ------------------------------------------------ Core download logic
    def download_from_channel(self, folder, url, limit, show_progress):
        """One yt-dlp run; archive.txt skips known videos; afterwards failures are un-archived."""
        os.makedirs(folder, exist_ok=True)
        vtt_dir = os.path.join(folder, "vtt_files")
        os.makedirs(vtt_dir, exist_ok=True)
        self._load_meta_state(folder)

        try:
            if not self.wait_for_network("the download"):
                return

            if self.opts.get('retry'):
                self._release_unfinished_from_archive(folder)

            self._seed_archive(folder, self.opts.get('retry', False))

            totals = {'ok': 0, 'nosubs': 0, 'failed': 0, 'attempted': 0}
            remaining = limit
            result = None

            # Like the old downloader: if the network drops mid-run, wait and run again.
            # Safe because archive.txt already records what finished.
            for attempt in range(NETWORK_RESUMES + 1):
                before, _ = self.load_archive_ids(folder)
                if show_progress and attempt == 0:
                    self.send_message(
                        'status', f"Downloading up to {limit} new videos ({len(before)} already in archive.txt)...")

                result = self._run_download(folder, vtt_dir, url, remaining, show_progress, before)
                part = self._settle_archive(folder, before, result)
                for key in totals:
                    totals[key] += part[key]
                remaining = limit - totals['ok'] - totals['nosubs']

                if (result['network_error'] and not result['rate_limited']
                        and not self.should_stop.is_set() and remaining > 0 and attempt < NETWORK_RESUMES):
                    self.send_message('log', "  Network problem during the download; waiting to resume...")
                    if self.wait_for_network("the current download"):
                        continue
                break

            if totals['attempted'] == 0 and not result['rate_limited']:
                msg = "No new videos found (everything is already in archive.txt)"
                self.send_message('status' if show_progress else 'log', msg if show_progress else "  " + msg)
            else:
                summary = (f"✓ {totals['ok']} subtitles saved, {totals['nosubs']} without subtitles, "
                           f"{totals['failed']} failed")
                self.send_message('status' if show_progress else 'log', summary if show_progress else "  " + summary)

            if result['rate_limited']:
                self.rate_limited = True
                self._set_cooldown()
                if result['rl_line']:
                    self.send_message('log', f"  429 detail: {result['rl_line'][:300]}")
                current_video = result.get('current_video') or {}
                video_url = current_video.get('webpage_url')
                if video_url:
                    self.send_message('log', f"  Video being processed: {video_url}")
                    self.send_message('set_url_if_empty', video_url)
                if totals['ok'] == 0:
                    cooldown_message = (
                        f"The app's {RATE_LIMIT_COOLDOWN_MIN}-minute cooldown is active. "
                        if RATE_LIMIT_COOLDOWN_MIN > 0 else
                        "Automatic cooldown is disabled. "
                    )
                    self.send_message(
                        'warning',
                        "The first subtitle request in this channel run was rejected (HTTP 429). This does not "
                        "show whether YouTube is throttling requests or rejecting this client/token, and it may "
                        "follow earlier requests from diagnostics or previous runs. "
                        + cooldown_message
                        + "After it expires, diagnose the exact video shown above before retrying.")
                else:
                    self.send_message(
                        'warning',
                        "YouTube returned HTTP 429 (rate limited). Stopped; videos without a transcript were "
                        "removed from archive.txt again so the next run retries them.")
        finally:
            self._flush_meta(folder)
            self._finalize_vtx_transcripts(folder)

    def _run_download(self, folder, vtt_dir, url, limit, show_progress, before):
        archive_file = os.path.join(folder, "archive.txt")

        command = self.base_args() + [
            "--download-archive", archive_file,
            "--force-write-archive",          # record even though nothing is "downloaded"; we verify afterwards
            "--max-downloads", str(limit),
            "--lazy-playlist",                # process entries as they arrive instead of listing the whole channel first
            "--skip-download",
            "--no-overwrites",                # an existing .vtt is never re-requested
            "--write-subs", "--write-auto-subs",
            "--sub-langs", SUB_LANGS,
            "--sub-format", "vtt",
            "--sleep-subtitles", str(SUBTITLE_SLEEP),
            "-P", vtt_dir,
            "-o", "%(title.:100)s [%(id)s].%(ext)s",
            "--print", "video:" + PRINT_TEMPLATE,
            # after_video = printed AFTER subtitles were written; --print implies --simulate,
            # so --no-simulate is required for files to be written at all.
            "--print", "after_video:" + PRINT_TEMPLATE,
            "--no-simulate",
        ]
        if self.opts.get('stop_first', STOP_AT_FIRST_ARCHIVED):
            command.append("--break-on-existing")
        command.append(url)

        result = {
            'ok': [], 'meta': {}, 'errors': [], 'rate_limited': False, 'rl_line': '',
            'violation': None, 'network_error': False, 'current_video': None,
        }

        try:
            proc, err_lines, thread = self._spawn(command, kill_on_rate_limit=True)
        except FileNotFoundError:
            self.send_message('error', "yt-dlp command not found.")
            return result
        except Exception as e:
            self.send_message('error', f"Failed to start yt-dlp: {e}")
            return result

        result['errors'] = err_lines
        watchdog = self._start_watchdog(proc, IDLE_TIMEOUT)

        try:
            for line in proc.stdout:
                if self.should_stop.is_set():
                    break
                self._last_activity = time.time()
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                vid = data.get('id')
                if not vid:
                    continue
                result['current_video'] = data

                # Tripwire: an archived video must never reach this point (yt-dlp should have
                # skipped it before fetching anything). If it does, stop right away.
                if vid in before:
                    result['violation'] = vid
                    self.send_message(
                        'error',
                        f"yt-dlp processed {vid} although it is already in archive.txt - archive is not "
                        f"being honored. Aborting to avoid wasted requests.")
                    self._stop_process(proc)
                    break

                result['meta'][vid] = data

                # Truth comes from the disk, not from yt-dlp having printed something.
                if self._has_vtt(vtt_dir, vid):
                    result['ok'].append(vid)
                    self._record_video(folder, data)
                    self.send_message('log', self.format_download_message(data))

                if show_progress:
                    self.send_message('progress', min(len(result['meta']) / limit, 1.0) * 100)
        finally:
            watchdog.set()
            if self.should_stop.is_set():
                self._stop_process(proc)
            self._finish(proc, thread)

        for line in err_lines:
            if self._has_marker(line, RATE_LIMIT_MARKERS):
                result['rate_limited'] = True
                result['rl_line'] = line
                break

        result['network_error'] = (
            not result['rate_limited']
            and self._has_marker("\n".join(err_lines), NETWORK_ERROR_MARKERS))

        return result

    def _settle_archive(self, folder, before, result):
        """
        yt-dlp archives a video even if its subtitle request failed. Check every video this run
        touched and take the ones without a transcript back OUT of archive.txt (unless it
        simply has no subtitles, or has failed MAX_FAILURES times).
        """
        after, _ = self.load_archive_ids(folder)
        ok = set(result['ok'])
        meta = result['meta']
        errs = result['errors']
        attempted = ((after - before) | set(meta)) - ok

        # Subtitle warnings from yt-dlp do NOT carry the video ID, so if any subtitle request
        # failed during this run we cannot tell which video it was: treat all as failed.
        run_sub_failure = any(
            'subtitle' in l.lower() and ('unable' in l.lower() or 'http error' in l.lower())
            for l in errs)

        nosubs, failed = [], {}
        for vid in sorted(attempted):
            related = " ".join(l for l in errs if vid in l)
            if vid not in meta:
                failed[vid] = related or "no data returned"
            elif (self._has_marker(related, FAIL_MARKERS) or run_sub_failure or result['rate_limited']):
                failed[vid] = related or "subtitle request failed"
            else:
                nosubs.append(vid)

        for vid in nosubs:
            self._record_video(folder, meta[vid])
            self.send_message('log', f"  No subtitles available: {meta[vid].get('title', vid)}")

        counts = self._load_failed(folder)
        for vid in list(ok) + nosubs:
            counts.pop(vid, None)

        release, gave_up = [], []
        for vid, why in failed.items():
            if (result['rate_limited'] or result.get('network_error')
                    or self._has_marker(why, NETWORK_ERROR_MARKERS)):
                release.append(vid)      # not the video's fault: retry without counting
                continue
            rec = counts.get(vid, {'count': 0})
            rec['count'] = rec.get('count', 0) + 1
            rec['last'] = datetime.now().isoformat()
            counts[vid] = rec
            (gave_up if rec['count'] >= MAX_FAILURES else release).append(vid)

        for vid in gave_up:
            if vid in meta:
                self._record_video(folder, meta[vid])
            self.send_message(
                'log', f"  Giving up on {vid} after {MAX_FAILURES} failures (kept in archive; tick Retry to try again)")

        to_unarchive = [v for v in release if v in after]
        if to_unarchive:
            self.remove_archive_entries(folder, to_unarchive, quiet=True)
        if release:
            self.send_message('log', f"  {len(release)} videos got no transcript and were un-archived (will retry next run)")

        self._save_failed(folder, counts)
        return {
            'ok': len(ok), 'nosubs': len(nosubs), 'failed': len(failed),
            'attempted': len(ok) + len(attempted),
        }

    def _seed_archive(self, folder, retry):
        """
        Make archive.txt cover everything we already have, so yt-dlp skips those videos
        before fetching anything: transcripts on disk, plus (normal mode) metadata.json entries.
        """
        known = self._ids_in_dir(os.path.join(folder, "vtx_files"), ('.vtx',))
        known |= self._ids_in_dir(os.path.join(folder, "vtt_files"), ('.vtt',))
        if not retry:
            _, metadata = self.load_metadata_ids(folder)
            known |= {e['id'] for e in metadata if self._is_video_metadata_entry(e)}

        archive_ids, _ = self.load_archive_ids(folder)
        missing = sorted(known - archive_ids)
        if missing:
            added = self.add_archive_entries(folder, missing, quiet=True)
            self.send_message(
                'log', f"  Added {added} known videos (on disk / in metadata) to archive.txt so yt-dlp skips them")

    def _release_unfinished_from_archive(self, folder):
        """Retry mode: un-archive videos that have no transcript on disk so yt-dlp picks them up again."""
        archive_ids, _ = self.load_archive_ids(folder)
        on_disk = self._ids_in_dir(os.path.join(folder, "vtx_files"), ('.vtx',))
        on_disk |= self._ids_in_dir(os.path.join(folder, "vtt_files"), ('.vtt',))
        stale = [v for v in archive_ids if v not in on_disk]
        if stale:
            removed = self.remove_archive_entries(folder, stale, quiet=True)
            self.send_message('log', f"  Retry mode: released {removed} archived videos that have no transcript on disk")
        counts = self._load_failed(folder)
        self._save_failed(folder, {k: v for k, v in counts.items() if k not in set(stale)})

    # Failure bookkeeping (counts per video so permanent failures stop being retried forever)
    def _failed_path(self, folder):
        return os.path.join(folder, "failed_ids.json")

    def _load_failed(self, folder):
        try:
            with open(self._failed_path(folder), 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_failed(self, folder, counts):
        try:
            with open(self._failed_path(folder), 'w', encoding='utf-8') as f:
                json.dump(counts, f, indent=2)
        except OSError:
            pass

    @staticmethod
    def _has_vtt(vtt_dir, vid):
        tag = f"[{vid}]"
        try:
            for name in os.listdir(vtt_dir):
                if tag in name and name.endswith(".vtt"):
                    if os.path.getsize(os.path.join(vtt_dir, name)) > 0:
                        return True
        except OSError:
            pass
        return False

    # -------------------------------------------- Known IDs / archive state
    @staticmethod
    def _ids_in_dir(path, exts):
        ids = set()
        if not os.path.isdir(path):
            return ids
        pattern = re.compile(r'\[([A-Za-z0-9_-]{11})\](?:\.[A-Za-z0-9_-]+)*$')
        try:
            names = os.listdir(path)
        except OSError:
            return ids
        for name in names:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in exts:
                continue
            m = pattern.search(stem)
            if m:
                ids.add(m.group(1))
        return ids

    @staticmethod
    def _is_video_metadata_entry(entry):
        """Distinguish video records from the minimal channel seed record."""
        if not isinstance(entry, dict) or not entry.get('id'):
            return False
        if entry.get('upload_date') or entry.get('duration') is not None:
            return True
        return '/watch?' in (entry.get('url') or '')

    def _load_meta_state(self, folder):
        _, metadata = self.load_metadata_ids(folder)
        self._meta_entries = list(metadata)
        self._meta_ids = {e.get('id') for e in metadata if isinstance(e, dict) and e.get('id')}

    def _flush_meta(self, folder):
        try:
            self.save_json(os.path.join(folder, "metadata.json"), self._meta_entries)
        except Exception as e:
            self.send_message('error', f"Failed to save metadata: {e}")

    def _record_video(self, folder, video_data):
        """Store metadata and archive entry for a verified video."""
        entry = self.create_metadata_entry(video_data)
        if not entry['id']:
            return
        if entry['id'] in self._meta_ids:
            for i, existing in enumerate(self._meta_entries):
                if isinstance(existing, dict) and existing.get('id') == entry['id']:
                    entry['original_timestamp'] = existing.get('original_timestamp') or existing.get('timestamp', '')
                    self._meta_entries[i] = entry
                    break
        else:
            self._meta_entries.append(entry)
            self._meta_ids.add(entry['id'])
        self.add_archive_entries(folder, [entry['id']], quiet=True)

    # -------------------------------------------------------- Finalization
    def _finalize_vtx_transcripts(self, folder):
        """Convert VTT -> VTX. Delete a VTT only when its VTX and a non-empty TXT exist."""
        vtt_dir = os.path.join(folder, "vtt_files")
        vtx_dir = os.path.join(folder, "vtx_files")
        txt_dir = os.path.join(folder, "txt_files")

        if not os.path.isdir(vtt_dir):
            return
        if not any(f.endswith('.vtt') for f in os.listdir(vtt_dir)):
            return

        os.makedirs(vtx_dir, exist_ok=True)

        try:
            result = VTTSegment().convert_vtt_directory_to_vtx(vtt_dir, vtx_dir)
        except Exception as e:
            self.send_message('error', f"VTT -> VTX conversion crashed: {e}")
            return

        if result.get('converted'):
            self.send_message('log', f"  Converted {len(result['converted'])} VTT files to VTX")
        if result.get('failed'):
            self.send_message('log', f"  Warning: {len(result['failed'])} VTT files failed to convert")

        deleted = 0
        if DELETE_VTT_AFTER_CONVERT:
            for vtt_file in result.get('converted', []):
                base = os.path.splitext(vtt_file)[0]
                vtx_file = os.path.join(vtx_dir, f"{base}.vtx")
                txt_file = os.path.join(txt_dir, f"{base}.txt")
                vtt_path = os.path.join(vtt_dir, vtt_file)
                try:
                    if (os.path.exists(vtx_file) and os.path.exists(txt_file)
                            and os.path.getsize(txt_file) > 0):
                        os.remove(vtt_path)
                        deleted += 1
                except Exception as e:
                    self.send_message('log', f"  Warning: Could not delete {vtt_file}: {e}")
        if deleted:
            self.send_message('log', f"  Deleted {deleted} verified VTT files")

        remaining = [f for f in os.listdir(vtt_dir) if f.endswith('.vtt')]
        if remaining:
            self.send_message('log', f"  Warning: {len(remaining)} unconfirmed VTT files remain in vtt_files/")

        if REMOVE_EMPTY_VTT_DIR and not os.listdir(vtt_dir):
            try:
                os.rmdir(vtt_dir)
            except OSError:
                pass

    # ---------------------------------------------------------- Metadata
    def create_metadata_entry(self, d, original_timestamp=None):
        def s(v):
            return "" if v is None else str(v).strip()

        def i(v):
            try:
                return int(v) if v else 0
            except (TypeError, ValueError):
                return 0

        cats = d.get('categories') or ['']
        entry = {
            'id': s(d.get('id')),
            'title': s(d.get('title')),
            'url': s(d.get('webpage_url')),
            'upload_date': s(d.get('upload_date')),
            'duration': i(d.get('duration')),
            'view_count': i(d.get('view_count')),
            'like_count': i(d.get('like_count')),
            'comment_count': i(d.get('comment_count')),
            'was_live': bool(d.get('was_live')),
            'is_live': bool(d.get('is_live')),
            'timestamp': datetime.now().isoformat(),
            'channel_name': s(d.get('channel') or d.get('uploader')),
            'channel_id': s(d.get('channel_id')),
            'channel_url': s(d.get('channel_url')),
            'subscriber_count': i(d.get('channel_follower_count')),
            'category': s(cats[0]),
        }
        if original_timestamp:
            entry['original_timestamp'] = s(original_timestamp)
        return entry

    def update_metadata(self, folder):
        """Refresh metadata older than 7 days (no subtitle downloads)."""
        metadata_file = os.path.join(folder, "metadata.json")
        if not os.path.exists(metadata_file):
            self.send_message('error', f"No previous downloads found. Directory: {folder}")
            return

        metadata = self.load_json(metadata_file)
        targets = [e for e in metadata
                   if self._is_video_metadata_entry(e) and self.is_timestamp_old(e.get('timestamp', ''), 7)]
        if not targets:
            self.send_message('status', "All metadata is up to date (updated within 7 days)")
            return

        def upload_key(item):
            d = item.get('upload_date', '')
            try:
                return datetime.strptime(d, '%Y%m%d') if len(d) == 8 else datetime.min
            except ValueError:
                return datetime.min

        targets.sort(key=upload_key, reverse=True)
        self.send_message('status', f"{len(targets)} videos need metadata updates")

        index = {e['id']: i for i, e in enumerate(metadata) if isinstance(e, dict) and e.get('id')}
        updated = 0
        batch_size = 25

        for start in range(0, len(targets), batch_size):
            if self.should_stop.is_set():
                break
            if not self.wait_for_network("the metadata update"):
                break

            batch = targets[start:start + batch_size]
            updated += self._update_batch(batch, metadata, index)
            self.save_json(metadata_file, metadata)
            self.send_message('progress', (min(start + batch_size, len(targets)) / len(targets)) * 100)

        if updated:
            self.send_message('status', f"✓ Updated metadata for {updated}/{len(targets)} videos")
        else:
            self.send_message('status', "No metadata was updated")

    def _update_batch(self, batch, metadata, index):
        originals = {v['id']: v for v in batch}
        urls = [f"https://www.youtube.com/watch?v={vid}" for vid in originals]
        command = self.base_args() + ["--print", PRINT_TEMPLATE] + urls

        try:
            proc, err_lines, thread = self._spawn(command)
        except Exception as e:
            self.send_message('error', f"Failed to start yt-dlp: {e}")
            return 0

        updated = 0
        try:
            for line in proc.stdout:
                if self.should_stop.is_set():
                    break
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                vid = data.get('id')
                old = originals.get(vid)
                if not old or vid not in index:
                    continue

                orig_ts = old.get('original_timestamp') or old.get('timestamp')
                entry = self.create_metadata_entry(data, original_timestamp=orig_ts)
                metadata[index[vid]] = entry
                updated += 1
                self.send_message('log', self.format_update_message(
                    entry, self.calculate_days_since(old.get('timestamp', ''))))
        finally:
            if self.should_stop.is_set():
                self._stop_process(proc)
            self._finish(proc, thread)

        if updated == 0 and err_lines:
            self.send_message('log', f"  yt-dlp said: {err_lines[0][:300]}")
        return updated

    def is_timestamp_old(self, timestamp_str, days=7):
        try:
            ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if ts.tzinfo:
                ts = ts.replace(tzinfo=None)
            return datetime.now() - ts > timedelta(days=days)
        except (ValueError, TypeError, AttributeError):
            return True

    # ------------------------------------------------- Consistency checks
    def check_channel(self, folder, channel_name):
        issues_found = issues_fixed = 0
        self.send_message('log', f"\n=== Checking {channel_name} ===")

        metadata_ids, metadata = self.load_metadata_ids(folder)
        archive_ids, duplicate_count = self.load_archive_ids(folder)
        vtx_ids = self.load_vtx_ids(folder)
        videos = [e for e in metadata if self._is_video_metadata_entry(e)]
        video_ids = {e['id'] for e in videos}

        # 1. duplicate archive lines
        if duplicate_count:
            issues_found += duplicate_count
            self.send_message('log', f"  Issue: Found {duplicate_count} duplicate archive entries")
            removed = self.remove_duplicate_archive_entries(folder)
            issues_fixed += removed
            archive_ids, _ = self.load_archive_ids(folder)

        # 2. archive entries without metadata
        orphaned_archive = [v for v in archive_ids if v not in metadata_ids]
        for v in orphaned_archive:
            self.send_message('log', f"  Issue: Archive entry without metadata: {v}")
        if orphaned_archive:
            issues_found += len(orphaned_archive)
            issues_fixed += self.remove_archive_entries(folder, orphaned_archive)
            archive_ids, _ = self.load_archive_ids(folder)

        # 3. metadata with a VTX but no archive entry
        missing_archive = []
        for e in videos:
            if e['id'] in vtx_ids and e['id'] not in archive_ids:
                missing_archive.append(e['id'])
                self.send_message('log', f"  Issue: Metadata with VTX but no archive entry: {e.get('title', e['id'])}")
        if missing_archive:
            issues_found += len(missing_archive)
            issues_fixed += self.add_archive_entries(folder, missing_archive)
            archive_ids, _ = self.load_archive_ids(folder)

        # 4. metadata with neither VTX nor archive (the channel seed record is left alone)
        orphaned_meta = []
        for e in videos:
            if e['id'] not in vtx_ids and e['id'] not in archive_ids:
                orphaned_meta.append(e['id'])
                self.send_message('log', f"  Issue: Metadata without archive or VTX: {e.get('title', e['id'])}")
        if orphaned_meta:
            issues_found += len(orphaned_meta)
            if self.remove_metadata_entries(folder, metadata, orphaned_meta):
                issues_fixed += len(orphaned_meta)

        # 5. informational
        missing_vtx = (video_ids & archive_ids) - vtx_ids
        if missing_vtx:
            self.send_message(
                'log',
                f"  Info: {len(missing_vtx)} videos have metadata/archive but no VTX "
                f"(tick 'Retry videos that have no transcript on disk' to re-fetch them)")

        if issues_found == 0:
            self.send_message('log', "  ✓ No issues found")
        return issues_found, issues_fixed

    def load_metadata_ids(self, folder):
        metadata_file = os.path.join(folder, "metadata.json")
        if not os.path.exists(metadata_file):
            return set(), []
        try:
            metadata = self.load_json(metadata_file)
            return {i['id'] for i in metadata if isinstance(i, dict) and i.get('id')}, metadata
        except Exception as e:
            self.send_message('log', f"  Error reading metadata: {e}")
            return set(), []

    def load_archive_ids(self, folder):
        archive_file = os.path.join(folder, "archive.txt")
        if not os.path.exists(archive_file):
            return set(), 0
        ids, dupes = set(), 0
        try:
            with open(archive_file, 'r', encoding='utf-8') as f:
                for line in f:
                    vid = self._parse_archive_video_id(line)
                    if not vid:
                        continue
                    if vid in ids:
                        dupes += 1
                    else:
                        ids.add(vid)
            return ids, dupes
        except Exception as e:
            self.send_message('log', f"  Error reading archive: {e}")
            return set(), 0

    def load_vtx_ids(self, folder):
        return self._ids_in_dir(os.path.join(folder, "vtx_files"), ('.vtx',))

    def remove_metadata_entries(self, folder, metadata, video_ids):
        try:
            ids = set(video_ids)
            cleaned = [m for m in metadata if not (isinstance(m, dict) and m.get('id') in ids)]
            self.save_json(os.path.join(folder, "metadata.json"), cleaned)
            self.send_message('log', f"  Fixed: Removed {len(ids)} orphaned metadata entries")
            return True
        except Exception as e:
            self.send_message('log', f"  Error saving metadata: {e}")
            return False

    def remove_archive_entries(self, folder, video_ids, quiet=False):
        archive_file = os.path.join(folder, "archive.txt")
        if not os.path.exists(archive_file):
            return 0
        targets = set(video_ids)
        try:
            with open(archive_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            kept, removed = [], 0
            for line in lines:
                if self._parse_archive_video_id(line) in targets:
                    removed += 1
                else:
                    kept.append(line)
            with open(archive_file, 'w', encoding='utf-8') as f:
                f.writelines(kept)
            if not quiet:
                self.send_message('log', f"  Fixed: Removed {removed} orphaned archive entries")
            return removed
        except Exception as e:
            self.send_message('log', f"  Error updating archive: {e}")
            return 0

    def add_archive_entries(self, folder, video_ids, quiet=False):
        """Append `youtube <id>` lines (yt-dlp archive format) for IDs not already present."""
        archive_file = os.path.join(folder, "archive.txt")
        try:
            existing, needs_newline = set(), False
            if os.path.exists(archive_file):
                with open(archive_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                needs_newline = bool(content) and not content.endswith("\n")
                for line in content.splitlines():
                    vid = self._parse_archive_video_id(line)
                    if vid:
                        existing.add(vid)

            new_ids = [v for v in video_ids if v not in existing]
            if new_ids:
                with open(archive_file, 'a', encoding='utf-8') as f:
                    if needs_newline:
                        f.write("\n")
                    for vid in new_ids:
                        f.write(f"youtube {vid}\n")

            if not quiet:
                self.send_message('log', f"  Fixed: Added {len(new_ids)} missing archive entries")
            return len(new_ids)
        except Exception as e:
            self.send_message('log', f"  Error adding archive entries: {e}")
            return 0

    def remove_duplicate_archive_entries(self, folder):
        archive_file = os.path.join(folder, "archive.txt")
        if not os.path.exists(archive_file):
            return 0
        try:
            with open(archive_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            cleaned, seen, removed = [], set(), 0
            for line in lines:
                vid = self._parse_archive_video_id(line)
                if vid and vid in seen:
                    removed += 1
                    continue
                if vid:
                    seen.add(vid)
                cleaned.append(line)
            if removed:
                with open(archive_file, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned)
                self.send_message('log', f"  Fixed: Removed {removed} duplicate archive entries")
            return removed
        except Exception as e:
            self.send_message('log', f"  Error deduplicating archive: {e}")
            return 0

    @staticmethod
    def _parse_archive_video_id(line):
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].lower() == 'youtube':
            return parts[1]
        return None

    # ------------------------------------------------------------ Helpers
    def get_input_folder(self):
        """<project>/data/input  (this file lives in <project>/core/)"""
        return os.path.join(_ROOT, "data", "input")

    def get_channel_folder(self, channel_id):
        return os.path.join(self.get_input_folder(), channel_id)

    def get_all_channels(self, input_folder):
        return [d for d in os.listdir(input_folder)
                if os.path.isdir(os.path.join(input_folder, d))]

    def extract_channel_id(self, url):
        patterns = [
            r'youtube\.com/c/([^/?&#]+)',
            r'youtube\.com/channel/([^/?&#]+)',
            r'youtube\.com/@([^/?&#]+)',
            r'youtube\.com/user/([^/?&#]+)',
        ]
        for pattern in patterns:
            m = re.search(pattern, url)
            if m:
                return m.group(1)
        m = re.search(r'[&?]list=([^&]+)', url)
        if m:
            return m.group(1)
        return "youtube_subtitles"

    def load_json(self, filepath):
        """Load JSON; fall back to the .backup file if the main file is corrupt."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            backup = filepath + '.backup'
            if not os.path.exists(backup):
                raise
            self.send_message('warning', f"Corrupted JSON in {os.path.basename(filepath)}, recovering from backup")
            with open(backup, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return self._normalize_entries(data) if isinstance(data, list) else data

    def save_json(self, filepath, data):
        """Atomic write: temp file -> verify -> copy old to .backup -> replace."""
        payload = self._normalize_entries(data) if isinstance(data, list) else data
        tmp = filepath + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            with open(tmp, 'r', encoding='utf-8') as f:
                json.load(f)
            if os.path.exists(filepath):
                try:
                    shutil.copy2(filepath, filepath + '.backup')
                except OSError:
                    pass
            os.replace(tmp, filepath)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    def _normalize_entries(self, data):
        """Keep dict entries that have an ID; drop duplicate IDs (first wins)."""
        seen, out, dupes = set(), [], 0
        for item in data:
            if not isinstance(item, dict) or not item.get('id'):
                continue
            if item['id'] in seen:
                dupes += 1
                continue
            seen.add(item['id'])
            out.append(item)
        if dupes:
            self.send_message('status', f"Removed {dupes} duplicate entries from metadata")
        return out

    # --------------------------------------------------------- Formatting
    def format_download_message(self, video_data):
        title = video_data.get('title') or 'Unknown'
        view_count = video_data.get('view_count') or 0
        cats = video_data.get('categories') or ['']
        category = cats[0] if cats else ''
        channel = video_data.get('channel') or video_data.get('uploader') or ''

        status = []
        if video_data.get('is_live'):
            status.append("LIVE")
        elif video_data.get('was_live'):
            status.append("Was Live")
        status_str = f" [{', '.join(status)}]" if status else ""

        message = f"Downloaded: {title}{status_str} ({self.format_date(video_data.get('upload_date', ''))})"
        if view_count:
            message += f" | Views: {self.format_number(view_count)}"
        if category:
            message += f" | Category: {category}"
        if channel:
            message += f" | Channel: {channel}"
        return message

    def format_update_message(self, video_data, days_old):
        title = video_data.get('title') or 'Unknown'
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

    @staticmethod
    def format_date(date_str):
        if not date_str or len(date_str) != 8:
            return date_str or ''
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    @staticmethod
    def format_number(number):
        if not number:
            return "0"
        number = int(number)
        if number >= 1_000_000_000:
            return f"{number / 1_000_000_000:.1f}B"
        if number >= 1_000_000:
            return f"{number / 1_000_000:.1f}M"
        if number >= 1_000:
            return f"{number / 1_000:.1f}K"
        return str(number)

    @staticmethod
    def calculate_days_since(timestamp_str):
        try:
            ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if ts.tzinfo:
                ts = ts.replace(tzinfo=None)
            return str((datetime.now() - ts).days)
        except (ValueError, TypeError, AttributeError):
            return "?"

    # ------------------------------------------------------ Message queue
    def send_message(self, msg_type, content):
        """Worker thread -> UI thread"""
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
        elif msg_type == 'set_url_if_empty':
            if not self.url_input.get().strip():
                self.url_input.insert(0, content)
        elif msg_type in ('warning', 'error'):
            self.output.insert(tk.END, f"[{msg_type.upper()}] {content}\n")
            self.output.see(tk.END)
            self.status_text.set(content)
        elif msg_type == 'done':
            self.enable_buttons()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    downloader = SubtitleDownloader(root)
    downloader.protocol("WM_DELETE_WINDOW", root.destroy)

    root.mainloop()

    # caffeinate -i python3 core/atlas_downloader.py
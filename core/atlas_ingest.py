#!/usr/bin/env python3
"""
Convert a folder of yt-dlp output into Atlas's expected layout.

Usage:
    python atlas_ingest.py <yt-dlp-folder> [channel-name]

Example:
    python atlas_ingest.py ~/Downloads/SomeChannel
    python atlas_ingest.py ~/Downloads/SomeChannel SomeChannel

What it does:
    1. Finds every *.info.json in the folder you point at.
    2. Finds the matching .srt / .vtt transcript for each video.
    3. Strips timestamps and duplicate caption lines from the transcript.
    4. Writes everything into data/input/<channel>/ in the layout Atlas wants.

Your original yt-dlp folder is left untouched.

Before running this, your yt-dlp download should have been done with:
    yt-dlp --skip-download --write-subs --write-auto-subs --sub-langs "en.*" \
           --convert-subs srt --write-info-json \
           -o "%(title)s [%(id)s].%(ext)s" \
           -P "/some/folder" "<channel-url>/videos"

The important part is that filenames contain the video ID in brackets,
like "Video Title [xeApql7zeSY].info.json".
"""

import json
import re
import sys
from pathlib import Path


# Both .srt and .vtt have timestamp lines that we want to throw away.
SRT_TIMESTAMP = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->")
VTT_TIMESTAMP = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->")

# Subtitle files sometimes contain inline formatting tags like <i> or <c.color>.
HTML_TAG = re.compile(r"<[^>]+>")


def strip_subtitles(filepath):
    """Read a subtitle file and return plain text with timestamps removed."""
    lines = []
    for raw_line in filepath.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()

        # Skip blank lines.
        if not line:
            continue

        # Skip timestamp lines like "00:00:12,500 --> 00:00:14,200".
        if SRT_TIMESTAMP.match(raw_line) or VTT_TIMESTAMP.match(raw_line):
            continue

        # Skip header lines that appear at the top of .vtt files.
        if line.upper().startswith(("WEBVTT", "NOTE", "STYLE", "KIND:", "LANGUAGE:")):
            continue

        # Skip the numeric counter lines that appear in .srt files.
        if line.isdigit():
            continue

        # Strip formatting tags and keep the visible text.
        lines.append(HTML_TAG.sub("", line))

    # Auto-generated captions often repeat the same line many times in a row.
    # We only keep a line when it differs from the previous one.
    deduped = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    return " ".join(deduped)


def find_transcript(folder, video_id):
    """Find the best subtitle file for a video in the given folder.

    Prefers human-written English subs, then auto-generated English subs,
    then anything else that matches the video ID.
    """
    subtitles = list(folder.glob("*[" + video_id + "]*.srt"))
    if not subtitles:
        subtitles = list(folder.glob("*[" + video_id + "]*.vtt"))
    if not subtitles:
        return None

    # Sort so the preferred version comes first.
    def preference(path):
        name = path.name.lower()
        if ".en." in name and "orig" not in name:
            return 0
        if ".en-orig." in name:
            return 1
        return 2

    subtitles.sort(key=preference)
    return subtitles[0]


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python atlas_ingest.py <yt-dlp-folder> [channel-name]")
        print()
        print("Example:")
        print("    python atlas_ingest.py ~/Downloads/SomeChannel")
        print("    python atlas_ingest.py ~/Downloads/SomeChannel SomeChannel")
        sys.exit(1)

    raw_folder = Path(sys.argv[1]).expanduser().resolve()

    if not raw_folder.is_dir():
        print("Error: that folder does not exist: " + str(raw_folder))
        sys.exit(1)

    # If the user did not give a channel name, use the folder's name.
    if len(sys.argv) == 3:
        channel_name = sys.argv[2]
    else:
        channel_name = raw_folder.name

    # Where the final files will go. This is relative to wherever the script
    # is run from, so run it from your Atlas project folder.
    output_folder = Path("data") / "input" / channel_name
    txt_folder = output_folder / "txt_files"
    txt_folder.mkdir(parents=True, exist_ok=True)

    print("Reading yt-dlp files from: " + str(raw_folder))
    print("Writing Atlas files to:    " + str(output_folder))
    print()

    metadata = []
    no_subs = 0
    empty_subs = 0

    info_files = sorted(raw_folder.rglob("*.info.json"))

    if not info_files:
        print("Error: no .info.json files found.")
        print("Did you download with --write-info-json?")
        sys.exit(1)

    for info_path in info_files:
        try:
            record = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception as error:
            print("  skipped " + info_path.name + ": " + str(error))
            continue

        video_id = record.get("id")
        if not video_id:
            continue

        subtitle = find_transcript(info_path.parent, video_id)
        if subtitle is None:
            no_subs += 1
            continue

        transcript = strip_subtitles(subtitle)
        if not transcript.strip():
            empty_subs += 1
            continue

        # The transcript filename needs the same bracketed video ID as the
        # info.json file so Atlas can match them up.
        txt_name = info_path.name.replace(".info.json", ".txt")
        (txt_folder / txt_name).write_text(transcript, encoding="utf-8")

        metadata.append(record)

    # Atlas expects metadata.json to be a list of per-video records.
    (output_folder / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Finished.")
    print("  transcripts written: " + str(len(metadata)))
    print("  metadata records:    " + str(len(metadata)))

    if no_subs:
        print("  videos with no subtitles: " + str(no_subs))
    if empty_subs:
        print("  transcripts that were empty after cleaning: " + str(empty_subs))

    print()
    print("You can now run Atlas on this channel:")
    print("    python atlas.py " + channel_name)


if __name__ == "__main__":
    main()
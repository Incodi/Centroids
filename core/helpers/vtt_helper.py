import os
import re


class VTTSegment:
    """Convert downloaded WebVTT subtitle files to Centroids text files."""

    TIMESTAMP_PATTERN = re.compile(r'^\d{2}:\d{2}:\d{2}\.\d{3}.*$')
    COUNTER_PATTERN = re.compile(r'^\d+$')
    KARAOKE_TAG_PATTERN = re.compile(r'<\d{2}:\d{2}:\d{2}\.\d{3}><c>')
    KARAOKE_CLOSE_PATTERN = re.compile(r'</c>')
    ALIGNMENT_PATTERN = re.compile(r'align:start position:\d+%')
    SPEAKER_LABEL_PATTERN = re.compile(r'^\s*[A-Z]+\s*\d*\s*:\s*')
    WHITESPACE_PATTERN = re.compile(r'\s+')

    def convert_single_vtt_to_vtx(self, vtt_path, vtx_path):
        try:
            try:
                with open(vtt_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
            except UnicodeDecodeError:
                with open(vtt_path, 'r', encoding='latin-1') as file:
                    lines = file.readlines()

            cleaned = []
            previous = None
            for line in lines:
                line = line.strip()
                if not line or line == 'WEBVTT' or line.startswith(
                    ('Kind:', 'Language:', 'NOTE', 'STYLE')
                ):
                    continue
                if self.TIMESTAMP_PATTERN.match(line) or self.COUNTER_PATTERN.match(line):
                    continue

                line = self._clean_line(line)
                if line and line != previous:
                    cleaned.append(line)
                    previous = line

            os.makedirs(os.path.dirname(vtx_path), exist_ok=True)
            with open(vtx_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(cleaned))
            return True
        except (OSError, UnicodeError):
            try:
                os.remove(vtx_path)
            except OSError:
                pass
            return False

    def _clean_line(self, line):
        line = self.KARAOKE_TAG_PATTERN.sub('', line)
        line = self.KARAOKE_CLOSE_PATTERN.sub('', line)
        line = self.ALIGNMENT_PATTERN.sub('', line)
        line = self.SPEAKER_LABEL_PATTERN.sub('', line)
        line = re.sub(r'<\s*/?\s*(c|color|b|i|u)\s*>', '', line)
        return self.WHITESPACE_PATTERN.sub(' ', line).strip()

    def convert_vtt_directory_to_vtx(self, vtt_dir, vtx_dir):
        result = {'converted': [], 'skipped': [], 'failed': []}
        if not os.path.exists(vtt_dir):
            return result

        os.makedirs(vtx_dir, exist_ok=True)
        for filename in os.listdir(vtt_dir):
            if not filename.endswith('.vtt'):
                continue

            base_name = os.path.splitext(filename)[0]
            vtx_path = os.path.join(vtx_dir, f'{base_name}.vtx')
            if os.path.exists(vtx_path):
                result['skipped'].append(filename)
                continue

            vtt_path = os.path.join(vtt_dir, filename)
            if self.convert_single_vtt_to_vtx(vtt_path, vtx_path):
                result['converted'].append(filename)
            else:
                result['failed'].append(filename)
        return result

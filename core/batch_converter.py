import os
import re

try:
    from .vtt_helper import VTTSegment
except ImportError:
    from vtt_helper import VTTSegment


def find_all_channels_with_vtt(root_dir):
    """Find all channel directories that contain vtt_files"""
    input_dir = os.path.join(root_dir, "data", "input")
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return []
    
    channels = []
    for item in os.listdir(input_dir):
        channel_path = os.path.join(input_dir, item)
        
        # Skip files and special directories
        if not os.path.isdir(channel_path) or item.startswith('.'):
            continue
        
        # Check if vtt_files directory exists
        vtt_dir = os.path.join(channel_path, "vtt_files")
        if os.path.exists(vtt_dir):
            vtt_files = [f for f in os.listdir(vtt_dir) if f.endswith('.vtt')]
            if vtt_files:
                channels.append({
                    'handle': item,
                    'path': channel_path,
                    'vtt_count': len(vtt_files)
                })
    
    return channels


def get_migration_status(channel_info):
    """
    Check migration status for a channel.
    vtx_files/ is the "already done" signal.
    
    Returns:
        dict with keys:
            'total_vtt': total VTT files in vtt_files/
            'existing_vtx': existing VTX files in vtx_files/
            'missing': VTT files that don't have VTX yet
            'missing_files': set of base names missing VTX
    """
    vtt_dir = os.path.join(channel_info['path'], "vtt_files")
    vtx_dir = os.path.join(channel_info['path'], "vtx_files")
    
    # Get all VTT files
    vtt_files = {os.path.splitext(f)[0] for f in os.listdir(vtt_dir) if f.endswith('.vtt')}
    
    # Get existing VTX files
    if os.path.exists(vtx_dir):
        vtx_files = {os.path.splitext(f)[0] for f in os.listdir(vtx_dir) if f.endswith('.vtx')}
    else:
        vtx_files = set()
    
    # Find missing conversions
    missing = vtt_files - vtx_files
    
    return {
        'total_vtt': len(vtt_files),
        'existing_vtx': len(vtx_files),
        'missing': len(missing),
        'missing_files': missing
    }


def migrate_channel(handle, delete_vtt=False):
    """
    Migrate a channel's VTT files to VTX and TXT format.
    
    Args:
        handle: Channel handle name
        delete_vtt: If True, delete VTT files after verified successful migration
    
    Returns:
        dict with migration results
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtt_dir = os.path.join(root, "data", "input", handle, "vtt_files")
    vtx_dir = os.path.join(root, "data", "input", handle, "vtx_files")
    txt_dir = os.path.join(root, "data", "input", handle, "txt_files")
    
    if not os.path.exists(vtt_dir):
        return {
            'status': 'no_vtt',
            'handle': handle,
            'message': 'No vtt_files directory found'
        }
    
    # Create vtx_dir and txt_dir if they don't exist
    os.makedirs(vtx_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    
    # Convert VTT to VTX
    converter = VTTSegment()
    vtx_result = converter.convert_vtt_directory_to_vtx(vtt_dir, vtx_dir)
    
    # Convert VTT to TXT
    txt_result = convert_vtt_directory_to_txt(vtt_dir, txt_dir)
    
    # Delete VTT files if requested and verified
    deleted_count = 0
    if delete_vtt:
        deleted_count = _verified_delete_vtt_files(vtt_dir, txt_dir, vtx_dir)
    
    # Remove empty vtt_files directory
    if delete_vtt and os.path.exists(vtt_dir):
        try:
            if not os.listdir(vtt_dir):
                os.rmdir(vtt_dir) # NOTE: FILES WILL BE DELETED IF THIS IS CALLED, SO BE CAREFUL
        except:
            pass
    
    return {
        'status': 'success',
        'handle': handle,
        'vtx_converted': len(vtx_result['converted']),
        'vtx_skipped': len(vtx_result['skipped']),
        'vtx_failed': len(vtx_result['failed']),
        'txt_converted': len(txt_result['converted']),
        'txt_skipped': len(txt_result['skipped']),
        'txt_failed': len(txt_result['failed']),
        'deleted': deleted_count,
        'failed_files': vtx_result['failed'] if vtx_result['failed'] else None
    }


def _skip_vtt_line(line):
    """Check if VTT line should be skipped"""
    return (not line or line == "WEBVTT" or 
            line.startswith(("Kind:", "Language:", "NOTE", "STYLE")) or 
            re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3}.*$', line) or
            re.match(r'^\d+$', line))

def _clean_vtt_line(line):
    """Clean a VTT line for TXT output"""
    line = line.replace('[&nbsp;__&nbsp;]', 'FUCK')
    line = re.sub(r'\[(?!&nbsp;__&nbsp;).*?\]', '', line)
    line = re.sub(r'<[^>]*>', '', line)
    line = re.sub(r'&gt;&gt;|>>|&gt;|&lt;', '', line)
    line = re.sub(r'align:start position:\d+%', '', line)
    line = re.sub(r'^\s*[A-Z]+\s*\d*\s*:\s*', '', line)
    return re.sub(r'\s+', ' ', line).strip()

def convert_single_vtt_to_txt(vtt_path, txt_path):
    """Convert a single VTT file to TXT format"""
    try:
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(vtt_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    
    cleaned = []
    prev = None
    
    for line in lines:
        line = line.strip()
        
        if _skip_vtt_line(line):
            continue
        
        line = _clean_vtt_line(line)
        
        if not line:
            continue
        
        # Avoid consecutive duplicates
        if line != prev and line:
            cleaned.append(line)
            prev = line
    
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(cleaned))
    
    return True

def convert_vtt_directory_to_txt(vtt_dir, txt_dir):
    """Convert all VTT files in a directory to TXT format"""
    if not os.path.exists(vtt_dir):
        return {'converted': [], 'skipped': [], 'failed': []}
    
    os.makedirs(txt_dir, exist_ok=True)
    
    vtt_files = [f for f in os.listdir(vtt_dir) if f.endswith('.vtt')]
    
    result = {
        'converted': [],
        'skipped': [],
        'failed': []
    }
    
    for vtt_file in vtt_files:
        base_name = os.path.splitext(vtt_file)[0]
        txt_file = f"{base_name}.txt"
        vtt_path = os.path.join(vtt_dir, vtt_file)
        txt_path = os.path.join(txt_dir, txt_file)
        
        # Skip if TXT already exists
        if os.path.exists(txt_path):
            result['skipped'].append(vtt_file)
            continue
        
        # Attempt conversion
        if convert_single_vtt_to_txt(vtt_path, txt_path):
            result['converted'].append(vtt_file)
        else:
            result['failed'].append(vtt_file)
    
    return result

def _verified_delete_vtt_files(vtt_dir, txt_dir, vtx_dir):
    """
    Delete VTT files only if both VTX and TXT counterparts exist and are non-empty.
    
    Args:
        vtt_dir: Path to vtt_files directory
        txt_dir: Path to txt_files directory
        vtx_dir: Path to vtx_files directory
    
    Returns:
        Number of VTT files deleted
    """
    if not os.path.exists(vtt_dir):
        return 0
    
    deleted = 0
    vtt_files = [f for f in os.listdir(vtt_dir) if f.endswith('.vtt')]
    
    for vtt_file in vtt_files:
        base_name = os.path.splitext(vtt_file)[0]
        txt_file = os.path.join(txt_dir, f"{base_name}.txt")
        vtx_file = os.path.join(vtx_dir, f"{base_name}.vtx")
        vtt_path = os.path.join(vtt_dir, vtt_file)
        
        # Only delete if both VTX and TXT exist and are non-empty
        if os.path.exists(vtx_file) and os.path.exists(txt_file):
            try:
                if os.path.getsize(txt_file) > 0 and os.path.getsize(vtx_file) > 0:
                    os.remove(vtt_path)
                    deleted += 1
            except:
                pass
    
    return deleted


def batch_migrate_all(delete_vtt=False, force_channels=None):
    """
    Migrate all channels from VTT to VTX format.
    
    Args:
        delete_vtt: If True, delete VTT files after verified migration
        force_channels: List of specific channel handles to migrate
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Find all channels with VTT files
    all_channels = find_all_channels_with_vtt(root)
    
    if not all_channels:
        print("❌ No channels with VTT files found in data/input/")
        return
    
    # Filter channels if specific ones requested
    if force_channels:
        channels_to_process = [ch for ch in all_channels if ch['handle'] in force_channels]
        if not channels_to_process:
            print(f"❌ None of the specified channels found: {force_channels}")
            return
    else:
        channels_to_process = all_channels
    
    print(f"\n{'='*70}")
    print(f"🎬 VTT TO VTX MIGRATION TOOL")
    print(f"{'='*70}")
    print(f"Found {len(all_channels)} channel(s) with VTT files")
    print(f"Processing {len(channels_to_process)} channel(s)")
    print(f"Delete VTT after migration: {'✓' if delete_vtt else '✗'}")
    print(f"{'='*70}\n")
    
    results = {
        'success': [],
        'no_vtt': [],
        'error': []
    }
    
    # Process each channel
    for i, channel in enumerate(channels_to_process, 1):
        handle = channel['handle']
        vtt_count = channel['vtt_count']
        
        print(f"[{i}/{len(channels_to_process)}] Processing: {handle} ({vtt_count} VTT files)")
        
        # Check migration status
        channel_info = {'path': os.path.join(root, "data", "input", handle)}
        status = get_migration_status(channel_info)
        print(f"  ℹ️  Existing VTX: {status['existing_vtx']}, Missing: {status['missing']}")
        
        # Migrate
        result = migrate_channel(handle, delete_vtt=delete_vtt)
        
        if result['status'] == 'success':
            print(f"✓ VTX: {result['vtx_converted']} converted, {result['vtx_skipped']} skipped, {result['vtx_failed']} failed")
            print(f"✓ TXT: {result['txt_converted']} converted, {result['txt_skipped']} skipped, {result['txt_failed']} failed")
            if delete_vtt:
                print(f"  Deleted {result['deleted']} VTT files (verified)")
            if result.get('failed_files'):
                print(f"  Failed files: {', '.join(result['failed_files'][:5])}")
                if len(result['failed_files']) > 5:
                    print(f"  ... and {len(result['failed_files']) - 5} more")
            results['success'].append(result)
        elif result['status'] == 'no_vtt':
            print(f"⊘ {result['message']}")
            results['no_vtt'].append(result)
        else:
            print(f"✗ Error: {result.get('message', 'Unknown error')}")
            results['error'].append(result)
        
        print()
    
    # Summary
    print()
    print(f"Channels successfully migrated: {len(results['success'])}")
    print(f"Channels with no VTT files: {len(results['no_vtt'])}")
    print(f"Channels with errors: {len(results['error'])}")
    
    total_vtx_converted = sum(r.get('vtx_converted', 0) for r in results['success'])
    total_txt_converted = sum(r.get('txt_converted', 0) for r in results['success'])
    total_deleted = sum(r.get('deleted', 0) for r in results['success'])
    print(f"\nTotal files converted to VTX: {total_vtx_converted}")
    print(f"Total files converted to TXT: {total_txt_converted}")
    if delete_vtt:
        print(f"Total VTT files deleted: {total_deleted}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate VTT files to VTX format for all channels')
    parser.add_argument('--delete-vtt', action='store_true',
                       help='Delete VTT files after verified migration (only if TXT exists and is non-empty)')
    parser.add_argument('--channels', nargs='+', 
                       help='Specific channel handles to migrate (space-separated)')
    
    args = parser.parse_args()
    
    batch_migrate_all(
        delete_vtt=args.delete_vtt,
        force_channels=args.channels
    )

# python3 centroids/core/batch_converter.py --delete-vtt
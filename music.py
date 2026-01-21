#!/usr/bin/env python3
"""
Classical Music Metadata Tagger using OpenRouter API
Analyzes FLAC filenames and generates proper classical music metadata
Automatically converts non-FLAC audio files (MP4/M4A/etc) to FLAC using ffmpeg

Requirements:
    pip install mutagen openai rich

Usage:
    export OPENROUTER_API_KEY="your-api-key-here"
    python3 music.py /path/to/your/music/folder
"""

import os
import sys
import json
import re
import subprocess
import shutil
from pathlib import Path
from mutagen.flac import FLAC
from openai import OpenAI

# Rich TUI imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.box import ROUNDED, DOUBLE
from rich.align import Align

console = Console()

# Default model - can be changed to any OpenRouter supported model
DEFAULT_MODEL = "google/gemini-3-flash-preview"

# Color scheme
COLORS = {
    "primary": "cyan",
    "secondary": "magenta",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "info": "blue",
    "muted": "dim white",
}


def show_banner():
    """Display the application banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ♪ ♫  [bold cyan]CLASSICAL MUSIC METADATA TAGGER[/bold cyan]  ♫ ♪                                  ║
║                                                                               ║
║   [dim]AI-Powered metadata extraction for your classical music collection[/dim]        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    console.print(banner)


def show_menu(folder_path=None):
    """Display the main menu and get user selection"""
    
    # Build subtitle with folder path if set
    subtitle = ""
    if folder_path:
        # Truncate long paths for display
        display_path = folder_path if len(folder_path) <= 50 else "..." + folder_path[-47:]
        subtitle = f"[dim]📁 {display_path}[/dim]"
    
    menu_panel = Panel(
        Align.center(
            Text.from_markup("""
[bold cyan]1[/bold cyan]  ▸  [white]Process New Files[/white]
    [dim]Tag files missing metadata, convert non-FLAC if needed[/dim]

[bold cyan]2[/bold cyan]  ▸  [white]Metadata Audit & Repair[/white]
    [dim]Review ALL files for consistency, regenerate if needed[/dim]

[bold cyan]3[/bold cyan]  ▸  [white]View Statistics[/white]
    [dim]Analyze your music library metadata coverage[/dim]

[bold cyan]4[/bold cyan]  ▸  [white]Settings[/white]
    [dim]Configure API model and preferences[/dim]

[bold cyan]5[/bold cyan]  ▸  [white]Change Folder[/white]
    [dim]Select a different music folder[/dim]

[bold red]Q[/bold red]  ▸  [white]Quit[/white]
            """),
            vertical="middle"
        ),
        title="[bold white]═══ MAIN MENU ═══[/bold white]",
        subtitle=subtitle if subtitle else None,
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 4),
    )
    console.print(menu_panel)
    
    choice = Prompt.ask(
        "\n[bold cyan]Select an option[/bold cyan]",
        choices=["1", "2", "3", "4", "5", "q", "Q"],
        default="1"
    )
    return choice.lower()


def setup_openrouter():
    """Initialize OpenRouter API client"""
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        console.print(Panel(
            "[bold red]Error:[/bold red] OPENROUTER_API_KEY environment variable not set\n\n"
            "[dim]Get your API key from:[/dim] [link=https://openrouter.ai/keys]https://openrouter.ai/keys[/link]\n\n"
            "[dim]Then set it with:[/dim] [cyan]export OPENROUTER_API_KEY='your-key-here'[/cyan]",
            title="[bold red]⚠ API Key Missing[/bold red]",
            border_style="red",
        ))
        sys.exit(1)
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client


def get_metadata_from_openrouter(client, filename, context_files=None, existing_metadata=None):
    """Use OpenRouter to parse filename and generate metadata"""
    
    context = ""
    if context_files:
        context = f"\n\nOther files in the same folder: {', '.join(context_files[:10])}"
    
    existing_context = ""
    if existing_metadata:
        existing_context = f"\n\nExisting metadata (may be incomplete/incorrect): {json.dumps(existing_metadata)}"
    
    prompt = f"""Analyze this classical music filename and extract metadata as JSON.

Filename: {filename}
{existing_context}

Return ONLY valid JSON with these fields (use null if uncertain):
{{
    "composer": "Last name, First name",
    "composer_short": "Last name only (e.g., Beethoven, Mozart, Chopin)",
    "work_full": "Full official work title with catalog number (e.g., Piano Concerto No. 2 in B-flat major, Op. 19)",
    "work_short": "SHORT searchable name that people actually search for (e.g., Piano Concerto No. 2, Symphony No. 5, Nocturne Op. 9 No. 2, Moonlight Sonata)",
    "movement": "Movement number and name if applicable (e.g., II. Adagio sostenuto)",
    "movement_name": "Just the movement name without number if exists (e.g., Adagio sostenuto)",
    "performers": ["Primary performer/conductor names"],
    "orchestra": "Orchestra/Ensemble name",
    "soloists": ["Soloist names with instrument"],
    "date": "Recording year if present",
    "disc": "Disc number if multi-disc",
    "track": "Track number (just the number, e.g., 01, 02, 12)",
    "suggested_filename": "MUST use work_short NOT work_full! Format: TrackNum - ComposerShort - WorkShort - Movement - Performer"
}}

CRITICAL Guidelines for suggested_filename:
- ALWAYS use the SHORT searchable work name (work_short), NOT the full official name!
- Users need to SEARCH for these files - use names people actually search for
- BAD: "01 - Beethoven - Piano Sonata No. 14 in C-sharp minor, Op. 27 No. 2 - I. Adagio sostenuto - Pollini" (TOO LONG!)
- GOOD: "01 - Beethoven - Moonlight Sonata - I. Adagio sostenuto - Pollini"
- BAD: "05 - Mozart - Piano Concerto No. 21 in C major, K. 467 - II. Andante - Uchida"
- GOOD: "05 - Mozart - Piano Concerto No. 21 - II. Andante - Uchida"
- Keep filenames SHORT and SEARCHABLE
- No catalog numbers (Op., K., BWV) in filename - put those in work_full only
- No key signatures in filename (C major, D minor, etc.)

Other guidelines:
- Composer should be "Last, First" format, composer_short is just "Last"
- work_full should include catalog numbers and key for metadata storage
- work_short should be what people type when searching (simple, memorable names)

{context}"""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        text = response.choices[0].message.content.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        elif text.startswith('```') and text.endswith('```'):
            text = text.strip('`').strip()
            if text.startswith('json'):
                text = text[4:].strip()
        
        metadata = json.loads(text)
        return metadata
    except Exception as e:
        console.print(f"  [red]✗ Error parsing with OpenRouter:[/red] {e}")
        return None


def validate_flac_file(file_path):
    """Check if file is a valid FLAC file"""
    try:
        # Check file header for FLAC signature
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'fLaC':
                return False, f"Not a FLAC file (header: {header[:20]})"
        
        # Try to open with mutagen
        audio = FLAC(file_path)
        return True, audio
    except Exception as e:
        return False, str(e)


def has_proper_metadata(audio):
    """Check if FLAC file has proper classical music metadata"""
    # Essential fields for classical music
    required_fields = {
        'composer': ['COMPOSER'],
        'work': ['ALBUM', 'WORK'],
        'title': ['TITLE'],
        'artist': ['ARTIST', 'ALBUMARTIST'],
    }
    
    missing = []
    for field_name, tag_options in required_fields.items():
        # Check if any of the tag options has a non-empty value
        has_value = False
        for tag in tag_options:
            if tag in audio and audio[tag] and audio[tag][0].strip():
                has_value = True
                break
        if not has_value:
            missing.append(field_name)
    
    if missing:
        return False, missing
    return True, []


def get_current_metadata(audio):
    """Extract current metadata from FLAC file for display"""
    metadata = {}
    fields = ['COMPOSER', 'ALBUM', 'WORK', 'TITLE', 'ARTIST', 'ALBUMARTIST', 
              'ORCHESTRA', 'ENSEMBLE', 'PERFORMER', 'DATE', 'DISCNUMBER', 'TRACKNUMBER']
    
    for field in fields:
        if field in audio and audio[field]:
            metadata[field] = audio[field][0] if len(audio[field]) == 1 else list(audio[field])
    
    return metadata


def detect_actual_format(file_path):
    """Detect the actual audio format based on file header"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)
            
            # Check for various audio format signatures
            if header[:4] == b'fLaC':
                return 'flac'
            elif header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                return 'wav'
            elif header[:3] == b'ID3' or header[:2] == b'\xff\xfb':
                return 'mp3'
            elif header[:4] == b'OggS':
                return 'ogg'
            elif header[4:8] == b'ftyp' or header[:4] == b'\x00\x00\x00\x18' or header[:4] == b'\x00\x00\x00\x1c' or header[:4] == b'\x00\x00\x00 ':
                # MP4/M4A container (ftyp box or size header)
                return 'm4a'
            else:
                return 'unknown'
    except Exception:
        return 'unknown'


def convert_to_flac(file_path):
    """Convert non-FLAC audio file to FLAC using ffmpeg"""
    file_path = Path(file_path)
    
    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        console.print("  [yellow]⚠ ffmpeg not found.[/yellow] Install with: [cyan]brew install ffmpeg[/cyan]")
        return None
    
    # Detect actual format for better messaging
    actual_format = detect_actual_format(file_path)
    console.print(f"  [blue]ℹ[/blue] Detected actual format: [bold]{actual_format.upper()}[/bold]")
    
    # Create temporary output file with a clearly different name
    temp_output = file_path.parent / f".{file_path.stem}_converted.flac"
    
    # Create backup folder on Desktop
    backup_folder = Path.home() / "Desktop" / "music_backups"
    backup_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        console.print(f"  [cyan]🔄 Converting[/cyan] {actual_format.upper()} → FLAC...")
        
        # Run ffmpeg to convert to FLAC
        # Use -loglevel error to reduce output noise
        result = subprocess.run([
            'ffmpeg', '-y', 
            '-loglevel', 'error',
            '-i', str(file_path),
            '-c:a', 'flac',
            '-compression_level', '8',
            str(temp_output)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            console.print(f"  [red]✗ ffmpeg conversion failed:[/red] {error_msg[:300]}")
            if temp_output.exists():
                temp_output.unlink()
            return None
        
        # Verify the temp output was created and is valid
        if not temp_output.exists():
            console.print(f"  [red]✗ ffmpeg did not produce output file[/red]")
            return None
        
        # Backup original file to Desktop folder
        # Use a more descriptive backup name with original format
        backup_name = f"{file_path.stem}_original_{actual_format}{file_path.suffix}"
        backup_path = backup_folder / backup_name
        # If file with same name exists, add a number
        counter = 1
        while backup_path.exists():
            backup_path = backup_folder / f"{file_path.stem}_original_{actual_format}_{counter}{file_path.suffix}"
            counter += 1
        shutil.move(str(file_path), str(backup_path))
        
        # Replace with converted file
        shutil.move(str(temp_output), str(file_path))
        
        console.print(f"  [green]✓ Converted successfully[/green] [dim](original saved to ~/Desktop/music_backups/)[/dim]")
        
        # Validate the new file
        is_valid, audio = validate_flac_file(file_path)
        if is_valid:
            return audio
        else:
            console.print(f"  [red]✗ Converted file still not valid:[/red] {audio}")
            return None
            
    except Exception as e:
        console.print(f"  [red]✗ Conversion error:[/red] {e}")
        if temp_output.exists():
            temp_output.unlink()
        return None


def sanitize_filename(filename):
    """Remove invalid characters from filename"""
    # Remove characters that are invalid in filenames
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    # Replace multiple spaces with single space
    filename = ' '.join(filename.split())
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename.strip()


def rename_file(file_path, new_filename):
    """Rename a file safely, handling conflicts"""
    file_path = Path(file_path)
    new_filename = sanitize_filename(new_filename)
    
    # Keep the original extension
    new_path = file_path.parent / f"{new_filename}{file_path.suffix}"
    
    # If it's the same name, no need to rename
    if new_path == file_path:
        return file_path, False
    
    # Handle filename conflicts by adding a number
    counter = 1
    while new_path.exists() and new_path != file_path:
        new_path = file_path.parent / f"{new_filename} ({counter}){file_path.suffix}"
        counter += 1
    
    try:
        file_path.rename(new_path)
        return new_path, True
    except Exception as e:
        console.print(f"  [red]✗ Error renaming file:[/red] {e}")
        return file_path, False


def apply_metadata_to_flac(file_path, metadata, audio=None, rename=True):
    """Apply metadata to FLAC file and optionally rename it"""
    try:
        # Use provided audio object or validate the file
        if audio is None:
            is_valid, result = validate_flac_file(file_path)
            if not is_valid:
                console.print(f"  [yellow]⚠ Invalid FLAC file:[/yellow] {result}")
                return False, file_path
            audio = result
        
        # Clear existing tags
        audio.clear()
        
        # Map metadata to FLAC tags
        if metadata.get('composer'):
            audio['COMPOSER'] = metadata['composer']
        
        # Use work_full for ALBUM/WORK, fallback to work_short or work
        work_full = metadata.get('work_full') or metadata.get('work_short') or metadata.get('work')
        work_short = metadata.get('work_short') or metadata.get('work_full') or metadata.get('work')
        
        if work_full:
            audio['ALBUM'] = work_full
            audio['WORK'] = work_full
        
        # Create a searchable TITLE that includes the work name
        # Format: "Work Short - Movement" for better searchability
        title_parts = []
        if work_short:
            title_parts.append(work_short)
        if metadata.get('movement'):
            title_parts.append(metadata['movement'])
        
        if title_parts:
            # Join with " - " for searchability
            searchable_title = ' - '.join(title_parts)
            audio['TITLE'] = searchable_title
        elif work_full:
            audio['TITLE'] = work_full
        
        # Also store movement separately if present
        if metadata.get('movement'):
            audio['MOVEMENT'] = metadata['movement']
        
        if metadata.get('performers'):
            performers = metadata['performers']
            if isinstance(performers, list):
                audio['ARTIST'] = ', '.join(performers)
                audio['ALBUMARTIST'] = ', '.join(performers)
            else:
                audio['ARTIST'] = performers
                audio['ALBUMARTIST'] = performers
        
        if metadata.get('orchestra'):
            audio['ORCHESTRA'] = metadata['orchestra']
            audio['ENSEMBLE'] = metadata['orchestra']
        
        if metadata.get('soloists'):
            soloists = metadata['soloists']
            if isinstance(soloists, list):
                audio['PERFORMER'] = soloists
            else:
                audio['PERFORMER'] = [soloists]
        
        if metadata.get('date'):
            audio['DATE'] = str(metadata['date'])
        
        if metadata.get('disc'):
            audio['DISCNUMBER'] = str(metadata['disc'])
        
        if metadata.get('track'):
            audio['TRACKNUMBER'] = str(metadata['track'])
        
        audio.save()
        
        # Rename file if suggested_filename is provided and rename is enabled
        new_path = file_path
        if rename and metadata.get('suggested_filename'):
            new_path, was_renamed = rename_file(file_path, metadata['suggested_filename'])
            if was_renamed:
                console.print(f"  [cyan]📝 Renamed to:[/cyan] {new_path.name}")
        
        return True, new_path
    except Exception as e:
        console.print(f"  [red]✗ Error writing metadata:[/red] {e}")
        return False, file_path



def display_metadata_table(metadata, title="Metadata"):
    """Display metadata in a nice table format"""
    table = Table(title=title, box=ROUNDED, border_style="cyan", show_header=True)
    table.add_column("Field", style="cyan", width=15)
    table.add_column("Value", style="white")
    
    for key, value in metadata.items():
        if value:
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            table.add_row(key, str(value))
    
    console.print(table)


def process_folder_normal(folder_path, client, dry_run=False):
    """Process all FLAC files in folder - normal mode (files missing metadata)"""
    folder = Path(folder_path)
    
    if not folder.exists():
        console.print(Panel(
            f"[red]Folder does not exist:[/red] {folder_path}",
            title="[bold red]Error[/bold red]",
            border_style="red"
        ))
        return
    
    # Get all FLAC files
    flac_files = list(folder.rglob('*.flac')) + list(folder.rglob('*.FLAC'))
    
    if not flac_files:
        console.print(Panel(
            f"No FLAC files found in [cyan]{folder_path}[/cyan]",
            title="[bold yellow]No Files[/bold yellow]",
            border_style="yellow"
        ))
        return
    
    # Summary panel
    mode_text = "[yellow]DRY RUN[/yellow] - No files will be modified" if dry_run else "[green]LIVE MODE[/green] - Files will be modified"
    console.print(Panel(
        f"[bold]Found:[/bold] {len(flac_files)} FLAC files\n"
        f"[bold]Mode:[/bold] {mode_text}\n"
        f"[bold]Path:[/bold] {folder_path}",
        title="[bold cyan]═══ PROCESSING NEW FILES ═══[/bold cyan]",
        border_style="cyan"
    ))
    console.print()
    
    processed = 0
    skipped = 0
    failed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing files...", total=len(flac_files))
        
        for i, file_path in enumerate(flac_files, 1):
            filename = file_path.name
            progress.update(task, description=f"[cyan]Processing:[/cyan] {filename[:40]}...")
            
            console.print(f"\n[bold][{i}/{len(flac_files)}][/bold] [white]{filename}[/white]")
            
            # Validate FLAC file before making API call
            is_valid, result = validate_flac_file(file_path)
            if not is_valid:
                console.print(f"  [yellow]⚠[/yellow] {result}")
                # Attempt to convert to FLAC
                audio = convert_to_flac(file_path)
                if audio is None:
                    console.print(f"  [dim]⏭ Skipping file[/dim]")
                    failed += 1
                    progress.advance(task)
                    continue
                result = audio  # Use the converted audio object
            
            # Check if file already has proper metadata
            has_metadata, missing_fields = has_proper_metadata(result)
            if has_metadata:
                console.print(f"  [green]✓[/green] Already has proper metadata")
                skipped += 1
                progress.advance(task)
                continue
            else:
                console.print(f"  [yellow]ℹ[/yellow] Missing: [dim]{', '.join(missing_fields)}[/dim]")
            
            # Get context from other files in same folder
            context_files = [f.name for f in file_path.parent.glob('*.flac')]
            
            # Get metadata from OpenRouter
            metadata = get_metadata_from_openrouter(client, filename, context_files)
            
            if metadata:
                display_metadata_table(metadata, title="Generated Metadata")
                
                if not dry_run:
                    success, new_path = apply_metadata_to_flac(file_path, metadata, result)
                    if success:
                        console.print("[green]  ✓ Metadata applied successfully[/green]")
                        processed += 1
                    else:
                        console.print("[red]  ✗ Failed to apply metadata[/red]")
                        failed += 1
                else:
                    console.print("[yellow]  ⏸ Dry run - not applied[/yellow]")
                    processed += 1
            else:
                console.print("[red]  ✗ Failed to generate metadata[/red]")
                failed += 1
            
            progress.advance(task)
    
    # Final summary
    console.print()
    summary_table = Table(title="Processing Summary", box=ROUNDED, border_style="green")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("[green]Processed[/green]", str(processed))
    summary_table.add_row("[cyan]Skipped (had metadata)[/cyan]", str(skipped))
    summary_table.add_row("[red]Failed[/red]", str(failed))
    summary_table.add_row("[bold]Total[/bold]", str(len(flac_files)))
    console.print(summary_table)


def process_folder_audit(folder_path, client, dry_run=False, auto_approve=False):
    """Audit ALL files for metadata consistency and correct if necessary"""
    folder = Path(folder_path)
    
    if not folder.exists():
        console.print(Panel(
            f"[red]Folder does not exist:[/red] {folder_path}",
            title="[bold red]Error[/bold red]",
            border_style="red"
        ))
        return
    
    # Get all FLAC files
    flac_files = list(folder.rglob('*.flac')) + list(folder.rglob('*.FLAC'))
    
    if not flac_files:
        console.print(Panel(
            f"No FLAC files found in [cyan]{folder_path}[/cyan]",
            title="[bold yellow]No Files[/bold yellow]",
            border_style="yellow"
        ))
        return
    
    # Summary panel
    mode_text = "[yellow]DRY RUN[/yellow] - No files will be modified" if dry_run else "[green]LIVE MODE[/green] - Files will be modified"
    console.print(Panel(
        f"[bold]Found:[/bold] {len(flac_files)} FLAC files\n"
        f"[bold]Mode:[/bold] {mode_text}\n"
        f"[bold]Path:[/bold] {folder_path}\n\n"
        "[dim]This mode reviews ALL files and uses AI to verify/fix metadata consistency.[/dim]",
        title="[bold magenta]═══ METADATA AUDIT & REPAIR ═══[/bold magenta]",
        border_style="magenta"
    ))
    console.print()
    
    verified = 0
    updated = 0
    skipped = 0
    failed = 0
    
    total_files = len(flac_files)
    
    for i, file_path in enumerate(flac_files, 1):
        filename = file_path.name
        
        # Progress indicator
        progress_pct = int((i / total_files) * 100)
        progress_bar = "━" * (progress_pct // 5) + "╺" + "─" * (20 - progress_pct // 5)
        console.print(f"\n[magenta]Progress:[/magenta] [{progress_bar}] {progress_pct}%")
        console.print(f"[bold][{i}/{total_files}][/bold] [white]{filename}[/white]")
        
        # Validate FLAC file
        is_valid, result = validate_flac_file(file_path)
        if not is_valid:
            console.print(f"  [yellow]⚠[/yellow] Invalid FLAC: {result}")
            # Attempt to convert to FLAC
            audio = convert_to_flac(file_path)
            if audio is None:
                console.print(f"  [dim]⏭ Skipping file[/dim]")
                failed += 1
                continue
            result = audio
        
        # Get current metadata
        current_metadata = get_current_metadata(result)
        
        if current_metadata:
            display_metadata_table(current_metadata, title="Current Metadata")
        else:
            console.print("  [dim]No existing metadata[/dim]")
        
        # Ask AI to analyze and potentially improve metadata
        context_files = [f.name for f in file_path.parent.glob('*.flac')]
        
        console.print("  [blue]🔍 Analyzing with AI...[/blue]")
        new_metadata = get_metadata_from_openrouter(client, filename, context_files, current_metadata)
        
        if new_metadata:
            # Check if metadata needs improvement
            changes_detected = False
            changes_summary = []
            
            # Get current and new values
            current_title = str(current_metadata.get('TITLE', '') or '')
            current_album = str(current_metadata.get('ALBUM', '') or '')
            current_composer = str(current_metadata.get('COMPOSER', '') or '')
            
            new_work_short = new_metadata.get('work_short') or ''
            new_movement = new_metadata.get('movement') or ''
            new_composer = new_metadata.get('composer') or ''
            
            # Build what the searchable title SHOULD be
            ideal_title_parts = []
            if new_work_short:
                ideal_title_parts.append(new_work_short)
            if new_movement:
                ideal_title_parts.append(new_movement)
            ideal_title = ' - '.join(ideal_title_parts) if ideal_title_parts else ''
            
            # Check if current TITLE is already in good searchable format
            # (contains work name, not just movement)
            current_title_has_work = False
            if new_work_short and current_title:
                # Check if current title contains key parts of work name
                work_keywords = [w.lower() for w in new_work_short.split() if len(w) > 3]
                title_lower = current_title.lower()
                matches = sum(1 for kw in work_keywords if kw in title_lower)
                if matches >= len(work_keywords) * 0.5:  # At least 50% of keywords match
                    current_title_has_work = True
            
            # Only suggest TITLE change if current title is missing the work name
            if not current_title_has_work and ideal_title and current_title.lower().strip() != ideal_title.lower().strip():
                changes_detected = True
                changes_summary.append(f"  • TITLE: [red]{current_title}[/red] → [green]{ideal_title}[/green]")
            
            # Check COMPOSER
            if new_composer and current_composer.lower().strip() != new_composer.lower().strip():
                # Only flag if current is empty or significantly different
                if not current_composer or current_composer.lower() not in new_composer.lower():
                    changes_detected = True
                    changes_summary.append(f"  • COMPOSER: [red]{current_composer}[/red] → [green]{new_composer}[/green]")
            
            # Check if file needs renaming (only if current name is worse than suggested)
            current_filename = file_path.stem
            suggested_filename = new_metadata.get('suggested_filename', '')
            
            if suggested_filename:
                # Check if current filename already has good searchable format
                current_has_work_in_name = False
                if new_work_short:
                    work_keywords = [w.lower() for w in new_work_short.split() if len(w) > 3]
                    filename_lower = current_filename.lower()
                    matches = sum(1 for kw in work_keywords if kw in filename_lower)
                    if matches >= len(work_keywords) * 0.5:
                        current_has_work_in_name = True
                
                # Only suggest rename if current filename lacks the work name
                if not current_has_work_in_name and suggested_filename != current_filename:
                    changes_detected = True
                    changes_summary.append(f"  • FILENAME: [red]{current_filename}[/red] → [green]{suggested_filename}[/green]")
            
            if changes_detected:
                console.print("  [yellow]⚠ Changes recommended:[/yellow]")
                for change in changes_summary:
                    console.print(change)
                
                display_metadata_table(new_metadata, title="Suggested Metadata")
                
                if not dry_run:
                    # Auto-approve or ask for confirmation
                    should_apply = auto_approve or Confirm.ask("  Apply these changes?", default=True)
                    if should_apply:
                        if auto_approve:
                            console.print("  [cyan]⚡ Auto-applying...[/cyan]")
                        success, new_path = apply_metadata_to_flac(file_path, new_metadata, result)
                        if success:
                            console.print("  [green]✓ Metadata updated[/green]")
                            updated += 1
                        else:
                            console.print("  [red]✗ Failed to update[/red]")
                            failed += 1
                    else:
                        console.print("  [dim]⏭ Skipped by user[/dim]")
                        skipped += 1
                else:
                    console.print("  [yellow]⏸ Dry run - not applied[/yellow]")
                    updated += 1
            else:
                console.print("  [green]✓ Metadata looks consistent[/green]")
                verified += 1
        else:
            console.print("  [red]✗ Failed to analyze with AI[/red]")
            failed += 1
    
    # Final summary
    console.print()
    summary_table = Table(title="Audit Summary", box=ROUNDED, border_style="magenta")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("[green]Verified OK[/green]", str(verified))
    summary_table.add_row("[yellow]Updated[/yellow]", str(updated))
    summary_table.add_row("[cyan]Skipped[/cyan]", str(skipped))
    summary_table.add_row("[red]Failed[/red]", str(failed))
    summary_table.add_row("[bold]Total[/bold]", str(len(flac_files)))
    console.print(summary_table)


def show_statistics(folder_path):
    """Display statistics about the music library"""
    folder = Path(folder_path)
    
    if not folder.exists():
        console.print(Panel(
            f"[red]Folder does not exist:[/red] {folder_path}",
            title="[bold red]Error[/bold red]",
            border_style="red"
        ))
        return
    
    flac_files = list(folder.rglob('*.flac')) + list(folder.rglob('*.FLAC'))
    
    if not flac_files:
        console.print(Panel(
            f"No FLAC files found",
            title="[bold yellow]No Files[/bold yellow]",
            border_style="yellow"
        ))
        return
    
    # Analyze files
    with_metadata = 0
    without_metadata = 0
    missing_fields_count = {}
    composers = set()
    total_size = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[blue]Analyzing library...", total=len(flac_files))
        
        for file_path in flac_files:
            is_valid, result = validate_flac_file(file_path)
            if is_valid:
                has_meta, missing = has_proper_metadata(result)
                if has_meta:
                    with_metadata += 1
                else:
                    without_metadata += 1
                    for field in missing:
                        missing_fields_count[field] = missing_fields_count.get(field, 0) + 1
                
                # Get composer
                if 'COMPOSER' in result and result['COMPOSER']:
                    composers.add(result['COMPOSER'][0])
                
                total_size += file_path.stat().st_size
            else:
                without_metadata += 1
            
            progress.advance(task)
    
    # Display stats
    console.print()
    
    # Main stats table
    stats_table = Table(title="Library Statistics", box=DOUBLE, border_style="blue")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right", style="white")
    
    stats_table.add_row("Total Files", str(len(flac_files)))
    stats_table.add_row("With Complete Metadata", f"[green]{with_metadata}[/green]")
    stats_table.add_row("Missing Metadata", f"[yellow]{without_metadata}[/yellow]")
    stats_table.add_row("Coverage", f"[{'green' if with_metadata/len(flac_files) > 0.8 else 'yellow'}]{with_metadata/len(flac_files)*100:.1f}%[/]")
    stats_table.add_row("Unique Composers", str(len(composers)))
    stats_table.add_row("Total Size", f"{total_size / (1024*1024*1024):.2f} GB")
    
    console.print(stats_table)
    
    # Missing fields breakdown
    if missing_fields_count:
        console.print()
        missing_table = Table(title="Missing Fields Breakdown", box=ROUNDED, border_style="yellow")
        missing_table.add_column("Field", style="yellow")
        missing_table.add_column("Missing In", justify="right")
        
        for field, count in sorted(missing_fields_count.items(), key=lambda x: x[1], reverse=True):
            missing_table.add_row(field, str(count))
        
        console.print(missing_table)
    
    # Top composers (if any)
    if composers:
        console.print()
        console.print(Panel(
            "\n".join(f"  • {c}" for c in sorted(list(composers))[:10]) + 
            (f"\n  [dim]...and {len(composers)-10} more[/dim]" if len(composers) > 10 else ""),
            title=f"[bold cyan]Composers ({len(composers)} total)[/bold cyan]",
            border_style="cyan"
        ))


def show_settings():
    """Display and modify settings"""
    global DEFAULT_MODEL
    
    settings_panel = Panel(
        f"""
[bold cyan]Current Settings[/bold cyan]

  [white]API Model:[/white] {DEFAULT_MODEL}
  [white]API Endpoint:[/white] https://openrouter.ai/api/v1

[dim]To change settings, edit the DEFAULT_MODEL variable in the script
or use environment variables.[/dim]
        """,
        title="[bold white]═══ SETTINGS ═══[/bold white]",
        border_style="cyan"
    )
    console.print(settings_panel)
    
    if Confirm.ask("\nChange model?", default=False):
        new_model = Prompt.ask("Enter new model name", default=DEFAULT_MODEL)
        DEFAULT_MODEL = new_model
        console.print(f"[green]✓ Model set to:[/green] {DEFAULT_MODEL}")


def main():
    # Interactive TUI mode
    console.clear()
    show_banner()
    
    # Check for command line folder path argument
    folder_path = None
    if len(sys.argv) >= 2 and not sys.argv[1].startswith('--'):
        folder_path = sys.argv[1]
        console.print(f"[green]✓ Folder path:[/green] {folder_path}\n")
    
    # Setup API client
    client = setup_openrouter()
    console.print("[green]✓ API client initialized[/green]\n")
    
    while True:
        choice = show_menu(folder_path)
        
        if choice == 'q':
            console.print(Panel(
                "[bold cyan]Thanks for using Classical Music Metadata Tagger![/bold cyan]\n\n"
                "[dim]♪ ♫ Happy listening! ♫ ♪[/dim]",
                border_style="cyan"
            ))
            break
        
        # Handle folder change
        if choice == '5':
            console.print()
            folder_path = Prompt.ask(
                "[cyan]Enter path to music folder[/cyan]",
                default=str(Path.home() / "Music")
            )
            console.clear()
            show_banner()
            console.print(f"[green]✓ Folder path:[/green] {folder_path}\n")
            continue
        
        # Get folder path if not set
        if choice in ['1', '2', '3'] and folder_path is None:
            console.print()
            folder_path = Prompt.ask(
                "[cyan]Enter path to music folder[/cyan]",
                default=str(Path.home() / "Music")
            )
        
        console.print()
        
        if choice == '1':
            # Process new files
            dry_run = Confirm.ask("Run in dry-run mode (preview only)?", default=False)
            process_folder_normal(folder_path, client, dry_run)
        
        elif choice == '2':
            # Metadata audit
            console.print(Panel(
                "[bold magenta]Metadata Audit Mode[/bold magenta]\n\n"
                "This will scan [bold]ALL[/bold] files in your library and use AI to:\n"
                "  • Verify existing metadata is correct and consistent\n"
                "  • Suggest corrections for incomplete or incorrect entries\n"
                "  • Standardize formatting (e.g., 'Last, First' for composers)\n"
                "  • Rename files to searchable format\n\n"
                "[yellow]You can choose to auto-approve all changes or review each one.[/yellow]",
                border_style="magenta"
            ))
            
            if Confirm.ask("Continue with audit?", default=True):
                dry_run = Confirm.ask("Run in dry-run mode (preview only)?", default=False)
                if not dry_run:
                    auto_approve = Confirm.ask("[cyan]Auto-approve all changes?[/cyan] (say Yes to apply all without prompting)", default=False)
                else:
                    auto_approve = False
                process_folder_audit(folder_path, client, dry_run, auto_approve)
        
        elif choice == '3':
            # Statistics
            show_statistics(folder_path)
        
        elif choice == '4':
            # Settings
            show_settings()
        
        console.print()
        Prompt.ask("[dim]Press Enter to continue...[/dim]", default="")
        console.clear()
        show_banner()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Git Contribution Analysis Script
Analyzes code contributions across all branches
"""

import subprocess
import re
from collections import defaultdict

def run_git_command(command):
    """Execute a git command and return the output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e}")
        return ""

def normalize_author(author_email):
    """Normalize author names by combining different email addresses"""
    # Map different emails to the same person
    email_map = {
        'fhufler@student.ethz.ch': 'Fynn Hufler',
        '117347762+fynnhufler@users.noreply.github.com': 'Fynn Hufler',
        'bentkesimeon@gmail.com': 'Simeon Bentke',
        'melissaballes2003@gmail.com': 'Melissa Balles'
    }
    
    # Extract email from "Name <email>" format
    email_match = re.search(r'<(.+?)>', author_email)
    if email_match:
        email = email_match.group(1)
        return email_map.get(email, author_email)
    return author_email

def get_line_statistics():
    """Get detailed line statistics for each author across all branches"""
    print("Analysiere Code-Beiträge über alle Branches...\n")
    
    # Get all unique authors
    authors_output = run_git_command(
        "git log --all --format='%an <%ae>' | sort -u"
    )
    
    stats = defaultdict(lambda: {
        'added': 0,
        'deleted': 0,
        'net': 0,
        'commits': 0,
        'files_changed': 0
    })
    
    # Process each author
    for author_line in authors_output.split('\n'):
        if not author_line:
            continue
            
        normalized_name = normalize_author(author_line)
        
        # Escape single quotes in author name for shell
        author_escaped = author_line.replace("'", "'\\''")
        
        # Get numstat for this author
        numstat_output = run_git_command(
            f"git log --all --author='{author_escaped}' --numstat --format="
        )
        
        # Get commit count
        commit_count = run_git_command(
            f"git log --all --author='{author_escaped}' --oneline | wc -l"
        )
        
        files_changed = set()
        
        for line in numstat_output.split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    added = int(parts[0]) if parts[0] != '-' else 0
                    deleted = int(parts[1]) if parts[1] != '-' else 0
                    filename = parts[2]
                    
                    stats[normalized_name]['added'] += added
                    stats[normalized_name]['deleted'] += deleted
                    stats[normalized_name]['net'] += (added - deleted)
                    files_changed.add(filename)
                except ValueError:
                    continue
        
        stats[normalized_name]['commits'] += int(commit_count)
        stats[normalized_name]['files_changed'] = len(files_changed)
    
    return stats

def get_file_type_statistics():
    """Get statistics per file type"""
    print("\nAnalysiere Beiträge nach Dateityp...\n")
    
    file_stats = defaultdict(lambda: defaultdict(lambda: {
        'added': 0,
        'deleted': 0
    }))
    
    # Get all authors
    authors_output = run_git_command(
        "git log --all --format='%an <%ae>' | sort -u"
    )
    
    for author_line in authors_output.split('\n'):
        if not author_line:
            continue
            
        normalized_name = normalize_author(author_line)
        
        # Escape single quotes in author name for shell
        author_escaped = author_line.replace("'", "'\\''")
        
        # Get numstat with filenames
        numstat_output = run_git_command(
            f"git log --all --author='{author_escaped}' --numstat --format="
        )
        
        for line in numstat_output.split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    added = int(parts[0]) if parts[0] != '-' else 0
                    deleted = int(parts[1]) if parts[1] != '-' else 0
                    filename = parts[2]
                    
                    # Get file extension
                    if '.' in filename:
                        ext = filename.split('.')[-1]
                    else:
                        ext = 'no_extension'
                    
                    file_stats[normalized_name][ext]['added'] += added
                    file_stats[normalized_name][ext]['deleted'] += deleted
                except ValueError:
                    continue
    
    return file_stats

def print_results(stats, file_stats):
    """Print formatted results"""
    print("="*80)
    print("GESAMT-STATISTIK ÜBER ALLE BRANCHES")
    print("="*80)
    print()
    
    # Sort by net lines (added - deleted)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['net'], reverse=True)
    
    total_added = sum(s['added'] for s in stats.values())
    total_deleted = sum(s['deleted'] for s in stats.values())
    total_net = sum(s['net'] for s in stats.values())
    total_commits = sum(s['commits'] for s in stats.values())
    
    print(f"{'Autor':<30} {'Commits':<10} {'Hinzugefügt':<12} {'Gelöscht':<12} {'Netto':<12} {'Dateien':<10} {'%':<8}")
    print("-"*100)
    
    for author, data in sorted_stats:
        percentage = (data['net'] / total_net * 100) if total_net > 0 else 0
        print(f"{author:<30} {data['commits']:<10} "
              f"+{data['added']:<11} -{data['deleted']:<11} "
              f"{data['net']:+<12} {data['files_changed']:<10} "
              f"{percentage:>6.1f}%")
    
    print("-"*100)
    print(f"{'TOTAL':<30} {total_commits:<10} "
          f"+{total_added:<11} -{total_deleted:<11} "
          f"{total_net:+<12}")
    print()
    
    # File type statistics
    print("="*80)
    print("BEITRÄGE NACH DATEITYP")
    print("="*80)
    print()
    
    for author in sorted_stats:
        author_name = author[0]
        if author_name in file_stats and file_stats[author_name]:
            print(f"\n{author_name}:")
            print(f"  {'Dateityp':<15} {'Hinzugefügt':<12} {'Gelöscht':<12} {'Netto':<12}")
            print("  " + "-"*51)
            
            sorted_files = sorted(
                file_stats[author_name].items(),
                key=lambda x: x[1]['added'] - x[1]['deleted'],
                reverse=True
            )
            
            for ext, data in sorted_files:
                net = data['added'] - data['deleted']
                print(f"  {ext:<15} +{data['added']:<11} -{data['deleted']:<11} {net:+<12}")

def get_recent_activity():
    """Show recent commit activity"""
    print("\n" + "="*80)
    print("LETZTE 20 COMMITS")
    print("="*80)
    print()
    
    commits = run_git_command(
        "git log --all --pretty=format:'%h | %an | %ad | %s' --date=short -20"
    )
    
    for commit in commits.split('\n'):
        if commit:
            parts = commit.split(' | ')
            if len(parts) >= 4:
                hash_id = parts[0]
                author = normalize_author(parts[1])
                date = parts[2]
                message = ' | '.join(parts[3:])
                print(f"{hash_id} | {author:<20} | {date} | {message}")

if __name__ == "__main__":
    print("Git Contributions Analyzer")
    print("="*80)
    print()
    
    # Get statistics
    stats = get_line_statistics()
    file_stats = get_file_type_statistics()
    
    # Print results
    print_results(stats, file_stats)
    
    # Show recent activity
    get_recent_activity()
    
    print("\n" + "="*80)
    print("Analyse abgeschlossen!")
    print("="*80)
import logging
from pathlib import Path
import time
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---
# These lists will be modified to create the 3 different filtering levels.
# This version represents the "base" configuration.

# EXCLUSION (Directory and Filename-based)
EXCLUDED_DIRS = {
    ".git", ".github", ".svn", ".idea", ".vscode",
    "__pycache__", "node_modules", "vendor", "release-notes", "changelog"

    "build", "bin", "obj", "dist", "out", "target", "third_party",
    "thirdparty", "license", "licenses",
}

EXCLUDED_FILENAME_ROOTS = {
    "license", "licenses", "copying", 
    "changelog", "contributing", "security",
    "code_of_conduct", "notice", "provider_information",
    "changes", "maintainers", "releasenotes"
}

EXCLUDED_EXACT_FILENAMES = {".gitignore", ".gitattributes"}

# REMOVED: The EXCLUDED_EXTENSIONS set is no longer needed.

# INCLUSION (File and Extension-based)
INCLUDED_FILES = {
    "CMakeLists.txt", "Makefile", "Kconfig", "west.yml", "pom.xml",
    "setup.py", "pyproject.toml", "requirements.txt",
    "Android.bp", "Android.mk",
}

INCLUDED_EXTENSIONS = {
    ".c", ".cpp", ".h", ".hpp", ".java", ".py",
    ".gradle", ".gradle.kts", ".dtsi", ".overlay", ".map", ".rc",
    ".proto", ".fbs", ".fidl", ".arxml", ".franca",
    ".md", ".rst", ".adoc", ".vspec",
}


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PathFilter:
    """
    A simplified filter that only checks file-level attributes.
    Directory-level filtering is now handled in the main walk.
    """
    def __init__(self):
        # MODIFIED: Removed self.excluded_extensions
        self.excluded_dirs = EXCLUDED_DIRS
        self.excluded_filename_roots = EXCLUDED_FILENAME_ROOTS
        self.excluded_exact_filenames = EXCLUDED_EXACT_FILENAMES
        self.included_files = INCLUDED_FILES
        self.included_extensions = INCLUDED_EXTENSIONS

    def _is_in_excluded_dir(self, path: Path) -> bool:
        """Checks if any part of the path is a generally excluded directory name."""
        return any(part.lower() in self.excluded_dirs for part in path.parts)

    def _is_excluded_by_filename(self, path: Path) -> bool:
        if path.name in self.excluded_exact_filenames: return True
        filename_lower = path.name.lower()
        for root in self.excluded_filename_roots:
            if (filename_lower == root or filename_lower.startswith(root + '.') or filename_lower.startswith(root + '-')):
                return True
        return False

    def is_included(self, path: Path) -> bool:
        """
        Determines if a file should be included in the final dataset.
        This function now uses a pure allow-list model for extensions.
        """
        # MODIFIED: Streamlined the logic by removing the check for excluded_extensions
        if self._is_in_excluded_dir(path): return False
        if self._is_excluded_by_filename(path): return False
        
        # Inclusion Check: The file is included if its name or extension is on the allow-list.
        if path.name in self.included_files: return True
        if path.suffix.lower() in self.included_extensions: return True
        
        return False

def main():
    start_time = time.time()
    
    load_dotenv()
    
    project_root_str = os.getenv('PROJECT_ROOT')
    if not project_root_str:
        logging.error("The 'PROJECT_ROOT' environment variable is not set.")
        return
        
    project_root = Path(project_root_str)
    repos_root = project_root / "repositories/all_repos"
    
    # MODIFIED: Added a version string to the output filename for clarity
    # This can be changed for each of the 3 versions.
    VERSION = "core"
    output_dir = project_root / "scripts" / "data_preparation" / "01_filtering"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'filtered_file_paths_{VERSION}.txt'
    
    excluded_repos_file = project_root / "repositories" / "test_repos.txt"

    if not repos_root.is_dir():
        logging.error(f"Repository root directory not found: '{repos_root}'")
        return

    excluded_repo_paths = set()
    if excluded_repos_file.is_file():
        logging.info(f"Loading excluded repositories from '{excluded_repos_file}'")
        with open(excluded_repos_file, 'r', encoding='utf-8') as f:
            excluded_repo_paths = {
                (repos_root / line.strip()).resolve() 
                for line in f if line.strip()
            }
        logging.info(f"Loaded {len(excluded_repo_paths)} repositories to exclude.")

    logging.info(f"Scanning for files in '{repos_root}'...")
    
    path_filter = PathFilter()
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            count = 0
            for root, dirs, files in os.walk(repos_root, topdown=True):
                current_dir = Path(root)
                
                dirs[:] = [
                    d for d in dirs 
                    if (current_dir / d).resolve() not in excluded_repo_paths
                ]
                
                for filename in files:
                    file_path = current_dir / filename
                    if path_filter.is_included(file_path):
                        f.write(f"{file_path}\n")
                        count += 1
        logging.info(f"Successfully filtered and saved {count:,} file paths to '{output_path}'.")
    except IOError as e:
        logging.error(f"Failed to write to output file '{output_path}': {e}")

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
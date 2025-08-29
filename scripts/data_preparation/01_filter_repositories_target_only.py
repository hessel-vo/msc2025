import logging
from pathlib import Path
import time
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---

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


# INCLUSION
INCLUDED_FILES = set()

INCLUDED_EXTENSIONS = {
    ".c", ".cpp", ".h", ".hpp", ".java", ".py",
}


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PathFilter:

    def __init__(self):
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
        """
        if self._is_in_excluded_dir(path): return False
        if self._is_excluded_by_filename(path): return False
        
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
    repos_root = project_root / "repositories" / "test_repos"
    
    VERSION = "target_only"
    output_dir = project_root / "scripts" / "data_preparation" / "01_filtering" / "repository_specific"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'filtered_file_paths_{VERSION}.txt'
    
    if not repos_root.is_dir():
        logging.error(f"Repository root directory not found: '{repos_root}'")
        return

    logging.info(f"Scanning for files in '{repos_root}'...")
    
    path_filter = PathFilter()
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            count = 0
            for root, dirs, files in os.walk(repos_root, topdown=True):
                current_dir = Path(root)
                
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
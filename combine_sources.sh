#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Usage and Argument Parsing
# -----------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [--output-file file] [--force] [--skip-examples] [--skip-tests]
          [--include-sub-dirs dir1,dir2,...] [--skip-cargo] [--skip-assets]
          [--skip-sql] [--skip-build-rs] [--workspace-crates crate1,crate2,...]
          [--include-python] [--python-only]

  --output-file       Path to the output file (default "all_source.txt")
  --force             Overwrite the output file if it already exists.
  --skip-examples     Exclude files from the "examples" directory.
  --skip-tests        Exclude files from the "tests" directory and any tests.rs files.
  --include-sub-dirs  Comma-separated list of subdirectory names to include for "src"
                      and workspace crates.
  --skip-cargo        Exclude Cargo.toml files.
  --skip-assets       Exclude .txt, .toml and .csv files from the "assets" directory.
  --skip-sql          Exclude .sql files.
  --skip-build-rs     Exclude build.rs file.
  --workspace-crates  Comma-separated list of workspace crate directories to include.
                      Defaults to "mcp-core,mcp-domain,mcp-protocol,mcp-client,mcp-server".
  --include-python    Include Python files (*.py) in all directories being searched and 
                      also include the ./py directory.
  --python-only       Only include Python files (*.py) in the output. Implies --include-python.
EOF
    exit 1
}

# Default values
OUTPUT_FILE="all_source.txt"
FORCE=0
SKIP_EXAMPLES=0
SKIP_TESTS=0
SKIP_ASSETS=0
SKIP_SQL=0
SKIP_BUILD_RS=0
INCLUDE_SUBDIRS=""
SKIP_CARGO=0
WORKSPACE_CRATES="mcp-core,mcp-domain,mcp-protocol,mcp-client,mcp-server"
INCLUDE_PYTHON=0
PYTHON_ONLY=0

# Use GNU getopt for long options
PARSED=$(getopt --options="" --long output-file:,force,skip-examples,skip-tests,include-sub-dirs:,skip-cargo,skip-assets,skip-sql,skip-build-rs,workspace-crates:,include-python,python-only -n "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    usage
fi
eval set -- "$PARSED"

while true; do
    case "$1" in
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --skip-examples)
            SKIP_EXAMPLES=1
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=1
            shift
            ;;
        --include-sub-dirs)
            INCLUDE_SUBDIRS="$2"
            shift 2
            ;;
        --skip-cargo)
            SKIP_CARGO=1
            shift
            ;;
        --skip-assets)
            SKIP_ASSETS=1
            shift
            ;;
        --skip-sql)
            SKIP_SQL=1
            shift
            ;;
        --skip-build-rs)
            SKIP_BUILD_RS=1
            shift
            ;;
        --workspace-crates)
            WORKSPACE_CRATES="$2"
            shift 2
            ;;
        --include-python)
            INCLUDE_PYTHON=1
            shift
            ;;
        --python-only)
            PYTHON_ONLY=1
            INCLUDE_PYTHON=1  # python-only implies include-python
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Split comma-separated values into arrays
IFS=',' read -r -a include_subdirs_arr <<< "$INCLUDE_SUBDIRS"
IFS=',' read -r -a workspace_crates_arr <<< "$WORKSPACE_CRATES"

# Remove empty include subdir if provided
if [[ "${include_subdirs_arr[*]}" == "" ]]; then
    include_subdirs_arr=()
fi

# -----------------------------------------------------------------------------
# Handle Output File Existence
# -----------------------------------------------------------------------------
if [[ -f "$OUTPUT_FILE" ]]; then
    if [[ $FORCE -eq 0 ]]; then
        echo "Error: File '$OUTPUT_FILE' already exists. Use --force to overwrite or remove it manually." >&2
        exit 1
    else
        rm -f "$OUTPUT_FILE" || { echo "Error: Could not remove '$OUTPUT_FILE'."; exit 1; }
    fi
fi

# -----------------------------------------------------------------------------
# Build the List of Directories
# -----------------------------------------------------------------------------
directories=("src")
if [[ $SKIP_EXAMPLES -eq 0 ]]; then
    directories+=("examples")
fi
if [[ $SKIP_TESTS -eq 0 ]]; then
    directories+=("tests")
fi
if [[ $SKIP_ASSETS -eq 0 ]]; then
    directories+=("assets")
fi
for crate in "${workspace_crates_arr[@]}"; do
    if [[ -n "$crate" ]]; then
        directories+=("$crate")
    fi
done

# Add py directory if include-python is enabled
if [[ $INCLUDE_PYTHON -eq 1 ]] && [[ -d "py" ]]; then
    directories+=("py")
fi

# -----------------------------------------------------------------------------
# Collect Rust Source Files, Cargo.toml Files, SQL Files, build.rs, Text/TOML/CSV Files, and Python Files
# -----------------------------------------------------------------------------
all_files=()

# Add Python files from the root directory if include-python is enabled
if [[ $INCLUDE_PYTHON -eq 1 ]]; then
    while IFS= read -r -d $'\0' file; do
        all_files+=("$file")
    done < <(find . -maxdepth 1 -type f -name "*.py" -print0)
fi

# Add the root Cargo.toml if it exists and if not skipped and not python-only
if [[ $SKIP_CARGO -eq 0 ]] && [[ $PYTHON_ONLY -eq 0 ]] && [[ -f "Cargo.toml" ]]; then
    all_files+=("Cargo.toml")
fi

# Add build.rs if it exists and if not skipped and not python-only
if [[ $SKIP_BUILD_RS -eq 0 ]] && [[ $PYTHON_ONLY -eq 0 ]] && [[ -f "build.rs" ]]; then
    all_files+=("build.rs")
fi

# Add SQL files from the root directory if not skipped and not python-only
if [[ $SKIP_SQL -eq 0 ]] && [[ $PYTHON_ONLY -eq 0 ]]; then
    while IFS= read -r -d $'\0' file; do
        all_files+=("$file")
    done < <(find . -maxdepth 1 -type f -name "*.sql" -print0)
fi

# Function to filter files based on allowed subdirectories
filter_files() {
    local dir="$1"
    local file
    
    # Build the find command parameters for file extensions
    local ext_params=()
    
    # If python-only, only search for Python files
    if [[ $PYTHON_ONLY -eq 1 ]]; then
        ext_params+=("-name" "*.py")
    else
        # Otherwise include all relevant file types
        ext_params+=("-name" "*.rs")
        
        # Add Python files if requested
        if [[ $INCLUDE_PYTHON -eq 1 ]]; then
            ext_params+=("-o" "-name" "*.py")
        fi
        
        # Add other file types if not skipped
        if [[ $SKIP_CARGO -eq 0 ]]; then
            ext_params+=("-o" "-name" "Cargo.toml")
        fi
        if [[ $SKIP_SQL -eq 0 ]]; then
            ext_params+=("-o" "-name" "*.sql")
        fi
    fi
    
    # Execute the find command with the built parameters
    while IFS= read -r -d $'\0' file; do
        # Remove the directory prefix and the following slash
        relative="${file#$dir/}"
        if [[ "$relative" != */* ]]; then
            echo "$file"
        else
            first_subdir="${relative%%/*}"
            for allowed in "${include_subdirs_arr[@]}"; do
                if [[ "$first_subdir" == "$allowed" ]]; then
                    echo "$file"
                    break
                fi
            done
        fi
    done < <(find "$dir" -type f \( "${ext_params[@]}" \) -print0)
}

# Check for build.rs in workspace crates if not skipped and not python-only
if [[ $SKIP_BUILD_RS -eq 0 ]] && [[ $PYTHON_ONLY -eq 0 ]]; then
    for crate in "${workspace_crates_arr[@]}"; do
        if [[ -n "$crate" && -f "$crate/build.rs" ]]; then
            all_files+=("$crate/build.rs")
        fi
    done
fi

# Process each directory
for dir in "${directories[@]}"; do
    if [[ -d "$dir" ]]; then
        if [[ "$dir" == "assets" && $PYTHON_ONLY -eq 0 ]]; then
            # For assets directory, find .txt, .toml, .csv, .sql and .py files (if not skipped)
            find_params=()
            find_params+=("-name" "*.txt" "-o" "-name" "*.toml" "-o" "-name" "*.csv")
            
            if [[ $SKIP_SQL -eq 0 ]]; then
                find_params+=("-o" "-name" "*.sql")
            fi
            
            if [[ $INCLUDE_PYTHON -eq 1 ]]; then
                find_params+=("-o" "-name" "*.py")
            fi
            
            while IFS= read -r -d $'\0' file; do
                all_files+=("$file")
            done < <(find "$dir" -type f \( "${find_params[@]}" \) -print0)
            
        # For "src" or a workspace crate and when allowed subdirectories are specified:
        elif [[ "$dir" == "src" || " ${workspace_crates_arr[*]} " == *" $dir "* || "$dir" == "py" ]] && [[ ${#include_subdirs_arr[@]} -gt 0 ]]; then
            while IFS= read -r file; do
                all_files+=("$file")
            done < <(filter_files "$dir")
        else
            # Build the find parameters
            find_params=()
            
            # If python-only, only search for Python files
            if [[ $PYTHON_ONLY -eq 1 ]]; then
                find_params+=("-name" "*.py")
            else
                # Otherwise include all relevant file types
                find_params+=("-name" "*.rs")
                
                # Add Python files if requested
                if [[ $INCLUDE_PYTHON -eq 1 ]]; then
                    find_params+=("-o" "-name" "*.py")
                fi
                
                # Add other file types if not skipped
                if [[ $SKIP_CARGO -eq 0 ]]; then
                    find_params+=("-o" "-name" "Cargo.toml")
                fi
                if [[ $SKIP_SQL -eq 0 ]]; then
                    find_params+=("-o" "-name" "*.sql")
                fi
            fi
            
            while IFS= read -r -d $'\0' file; do
                all_files+=("$file")
            done < <(find "$dir" -type f \( "${find_params[@]}" \) -print0)
        fi
    fi
done

# Remove duplicates and (if requested) filter out tests.rs files
temp_file=$(mktemp)
for file in "${all_files[@]}"; do
    if [[ $SKIP_TESTS -eq 1 ]] && [[ "$(basename "$file")" == "tests.rs" ]]; then
        continue
    fi
    echo "$file"
done | sort -u > "$temp_file"
mapfile -t all_files < "$temp_file"
rm "$temp_file"

# -----------------------------------------------------------------------------
# Generate and Write the Tree Structure
# -----------------------------------------------------------------------------
# A recursive function that writes a tree structure for the given directory.
# Parameters: current directory (relative path) and an indent string.
write_tree_structure() {
    local current_path="$1"
    local indent="$2"
    local subdirs=()
    local files=()
    local file

    # Iterate over all_files to pick out those inside current_path
    for file in "${all_files[@]}"; do
        if [[ "$file" == "$current_path/"* ]]; then
            relpath="${file#$current_path/}"
            if [[ "$relpath" != */* ]]; then
                files+=("$(basename "$file")")
            else
                subdir="${relpath%%/*}"
                if [[ ! " ${subdirs[*]} " =~ " $subdir " ]]; then
                    subdirs+=("$subdir")
                fi
            fi
        fi
    done

    # Write subdirectories first
    IFS=$'\n' sorted_subdirs=($(sort <<<"${subdirs[*]}"))
    unset IFS
    for sub in "${sorted_subdirs[@]}"; do
        echo "${indent}+-- ${sub}/" >> "$OUTPUT_FILE"
        write_tree_structure "$current_path/$sub" "${indent}|   "
    done

    # Custom order for files depending on python-only mode
    if [[ $PYTHON_ONLY -eq 1 ]]; then
        # For python-only mode, we only need to list Python files
        local py_files=()
        for f in "${files[@]}"; do
            if [[ "$f" == *.py ]]; then
                py_files+=("$f")
            fi
        done
        IFS=$'\n' sorted_py_files=($(sort <<<"${py_files[*]}"))
        unset IFS
        
        for f in "${sorted_py_files[@]}"; do
            echo "${indent}+-- $f" >> "$OUTPUT_FILE"
        done
    else
        # For normal mode, use the original ordering logic
        local cargo_file=""
        local build_file=""
        local mod_file=""
        local lib_file=""
        local others=()
        for f in "${files[@]}"; do
            case "$f" in
                Cargo.toml) cargo_file="$f" ;;
                build.rs) build_file="$f" ;;
                mod.rs) mod_file="$f" ;;
                lib.rs) lib_file="$f" ;;
                *) others+=("$f") ;;
            esac
        done
        IFS=$'\n' sorted_others=($(sort <<<"${others[*]}"))
        unset IFS

        [[ -n "$cargo_file" ]] && echo "${indent}+-- $cargo_file" >> "$OUTPUT_FILE"
        [[ -n "$build_file" ]] && echo "${indent}+-- $build_file" >> "$OUTPUT_FILE"
        [[ -n "$mod_file" ]] && echo "${indent}+-- $mod_file" >> "$OUTPUT_FILE"
        [[ -n "$lib_file" ]] && echo "${indent}+-- $lib_file" >> "$OUTPUT_FILE"
        for f in "${sorted_others[@]}"; do
            echo "${indent}+-- $f" >> "$OUTPUT_FILE"
        done
    fi
}

{
    echo "Project Structure:"
    # Root level files
    if [[ $PYTHON_ONLY -eq 0 ]]; then
        # Only show non-Python files if not in python-only mode
        if [[ $SKIP_CARGO -eq 0 ]] && [[ -f "Cargo.toml" ]]; then
            echo "+-- Cargo.toml"
        fi
        
        if [[ $SKIP_BUILD_RS -eq 0 ]] && [[ -f "build.rs" ]]; then
            echo "+-- build.rs"
        fi
        
        # List SQL files from root directory in the tree structure
        if [[ $SKIP_SQL -eq 0 ]]; then
            for file in "${all_files[@]}"; do
                if [[ "$file" == ./*.sql ]]; then
                    echo "+-- $(basename "$file")"
                fi
            done
        fi
    fi
    
    # Always include Python files from root if requested
    if [[ $INCLUDE_PYTHON -eq 1 ]]; then
        for file in "${all_files[@]}"; do
            if [[ "$file" == ./*.py ]]; then
                echo "+-- $(basename "$file")"
            fi
        done
    fi
    
    # Directories
    for dir in "${directories[@]}"; do
        if [[ -d "$dir" ]]; then
            # Check if there are any matching files in this directory before including it
            found_files=0
            for file in "${all_files[@]}"; do
                if [[ "$file" == "$dir/"* ]]; then
                    found_files=1
                    break
                fi
            done
            
            if [[ $found_files -eq 1 ]]; then
                echo "+-- ${dir}/"
                write_tree_structure "$dir" "    "
            fi
        fi
    done
    echo ""
    echo "===================================================="
    echo "====================  FILES  ======================="
    echo "===================================================="
    echo ""
} >> "$OUTPUT_FILE"

# -----------------------------------------------------------------------------
# Process and Write Each File's Contents
# -----------------------------------------------------------------------------
# For each file, write a begin delimiter line, then its contents, then an end delimiter line.
for file in "${all_files[@]}"; do
    # Here, file paths are already relative to the project root.
    relative_path="$file"
    {
        echo "=== BEGIN FILE: $relative_path ==="
        if [[ "$(basename "$file")" == "Cargo.toml" ]]; then
            cat "$file"
        elif [[ "$(basename "$file")" == "build.rs" ]]; then
            cat "$file"
        elif [[ "$file" == "assets/"* ]] && { [[ "$file" == *.txt ]] || [[ "$file" == *.toml ]] || [[ "$file" == *.csv ]] || [[ "$file" == *.sql ]] || [[ "$file" == *.py ]]; }; then
            # For text, TOML, CSV, SQL, and Python files in assets, just include the content
            cat "$file"
        elif [[ "$file" == *.sql ]]; then
            # For SQL files, just include the content
            cat "$file"
        else
            # For all other files (including Python and Rust), just include the content
            cat "$file"
        fi
        echo "=== END FILE: $relative_path ==="
        echo ""
    } >> "$OUTPUT_FILE"
done

echo "Generated $OUTPUT_FILE"
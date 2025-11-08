# System Commands (Darwin/macOS)

This project is being developed on **Darwin** (macOS). Here are the relevant system commands:

## File System Navigation

### Basic Commands
```bash
ls          # List directory contents
ls -la      # List all files including hidden, with details
cd <dir>    # Change directory
pwd         # Print working directory
mkdir <dir> # Create directory
rm <file>   # Remove file
rm -rf <dir> # Remove directory recursively
```

### macOS-Specific Notes
- Case-insensitive filesystem by default (though case-preserving)
- Hidden files start with `.` (like `.env`, `.gitignore`)
- Use `open .` to open current directory in Finder
- Use `open <file>` to open file with default application

## File Operations

### Reading Files
```bash
cat <file>           # Display entire file
head -n 20 <file>    # First 20 lines
tail -n 20 <file>    # Last 20 lines
less <file>          # Page through file
```

### Searching Files
```bash
find . -name "*.py"                    # Find Python files
find . -type f -name "test_*.py"       # Find test files
grep -r "pattern" .                     # Search for pattern recursively
grep -r "pattern" --include="*.py" .   # Search only in Python files
```

## Git Commands

### Basic Git Operations
```bash
git status                    # Check status
git branch                    # List branches
git checkout -b <branch>      # Create and switch to new branch
git add <file>                # Stage file
git add .                     # Stage all changes
git commit -m "message"       # Commit changes
git push origin <branch>      # Push to remote
git pull                      # Pull latest changes
git diff                      # Show unstaged changes
git diff --staged             # Show staged changes
git log                       # View commit history
git log --oneline             # Compact commit history
```

### Current Repository Info
- Current branch: `main`
- Main branch for PRs: `main`

## Process Management

```bash
ps aux                 # List all running processes
ps aux | grep python   # Find Python processes
kill <PID>             # Terminate process
kill -9 <PID>          # Force kill process
```

## Environment Variables

### View Environment
```bash
env                    # List all environment variables
echo $PATH             # Show PATH variable
echo $OPENAI_API_KEY   # Show specific variable
```

### Set Environment Variables
```bash
export VAR_NAME=value                    # Set for current session
export OPENAI_API_KEY="sk-..."          # Example
```

### Permanent Environment Variables
For permanent variables, add to `~/.zshrc` or `~/.bash_profile`:
```bash
echo 'export VAR_NAME=value' >> ~/.zshrc
source ~/.zshrc
```

## Docker Commands (if applicable)

```bash
docker ps                           # List running containers
docker ps -a                        # List all containers
docker-compose up                   # Start services
docker-compose up -d                # Start in background
docker-compose down                 # Stop services
docker-compose logs -f              # Follow logs
docker-compose ps                   # List compose services
```

## Network Commands

```bash
curl <url>                          # Make HTTP request
curl -I <url>                       # Get headers only
ping <host>                         # Check connectivity
netstat -an | grep LISTEN           # Show listening ports
lsof -i :<port>                     # See what's using a port
```

## Permissions

```bash
chmod +x <file>        # Make file executable
chmod 644 <file>       # Set file permissions (rw-r--r--)
chmod 755 <file>       # Set file permissions (rwxr-xr-x)
chown <user> <file>    # Change file owner
```

## Useful Utilities

### Text Processing
```bash
wc -l <file>           # Count lines
wc -w <file>           # Count words
sort <file>            # Sort lines
uniq <file>            # Remove duplicates
awk '{print $1}' <file> # Print first column
sed 's/old/new/g' <file> # Replace text
```

### Archives
```bash
tar -czf archive.tar.gz <dir>    # Create tar.gz archive
tar -xzf archive.tar.gz           # Extract tar.gz archive
zip -r archive.zip <dir>          # Create zip archive
unzip archive.zip                 # Extract zip archive
```

## macOS-Specific Commands

```bash
pbcopy < <file>        # Copy file contents to clipboard
pbpaste > <file>       # Paste clipboard to file
caffeinate             # Prevent system sleep
say "text"             # Text-to-speech
```

## Python/UV Specific

```bash
which python3          # Find Python executable location
python3 --version      # Check Python version
uv --version           # Check UV version
uv run python          # Run Python with UV
uv pip list            # List installed packages
```

## Development Workflow Integration

For this project, you'll commonly use:
```bash
# Navigate to project
cd /Users/lvarming/it-setup/projects/graphiti

# Check git status
git status

# Run development checks
make check

# Search for code patterns
grep -r "def add_episode" graphiti_core/

# Find specific files
find . -name "graphiti.py"
```

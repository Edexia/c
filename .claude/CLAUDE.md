For pointers/docs about how to interact with the edf/egf file formats, refer to:
- https://raw.githubusercontent.com/10decikelvin/edf/refs/heads/main/DOCS/SDK.md
- https://raw.githubusercontent.com/10decikelvin/egf/refs/heads/main/DOCS/SDK.md

For pointers/docs about the type of CLI that you should mimic, refer to:
- https://raw.githubusercontent.com/10decikelvin/edf/refs/heads/main/DOCS/CLI.md
- https://raw.githubusercontent.com/10decikelvin/egf/refs/heads/main/DOCS/CLI.md

You must use uv in this project. This project is a cli tool (installable trivially called c) that takes as command line arguments a list of egf files and outputs a self contained .html containing all necessary information that can be opened. If you ever need test egf/edf files, there are some in secret/. Don't move them out of the repo
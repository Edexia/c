Your job is to 0-1 this repo and implement the vision. The input format should still predominantly be the same as old. The way calculations are made should still be the same as old. However, you should make it so that instead of hacky unofficial GT data sourced from within egf files, the cli tool should scan the cwd for all edf files, look for the one with the correct hash, cache the filename (so future lookups are faster, but this cache can be invalidated). Do not trust comments in the old/ folder, there many major inaccuracies.


Edits:
- remove the essay based comparison matrix, only keep qwk-based comaprison matrixone
- for the qwk bars, instead of one for every noise variant, it should be a command line flag to set the noise assumption (default to the expectation one). Instead, the 3 qwk bars should be GT noise only, grading noise only, and essay sampling variance.


Ideal usage pattern is:

c <egf file> <folder with egf files> -> checks that it has all the edf's, if some are missing drop and warn -> 1. if singular one, just run the qwk interval calcs 2. if multiple ones but not all datasets are the same, then run the qwk intervals for each one 3. if multiple ones and all the datasets are the same, then run qwk examples and the NxN comparison matrix.

Another usage pattern is:

c --watch --base=<egf file> -> whenever a new egf file comes into the directory, it checks if it is comparabl (from same dataset) -> if from same dataset, generate the output html (send file path to terminal), and the expected P(new egf > old egf). If not, skip
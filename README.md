# tellurics-begone
Script for removing telluric features from astronomical spectra.

## Dependencies
numpy, scipy, matplotlib

## Usage
ensure ```tellurics_begone.py``` is in the same directory as ```telluric_models/```

import ```remove_tellurics``` from ```tellurics_begone.py``` and use in a script, returns the corrected spectrum as an array

or

run ```tellurics_begone.py``` in command line, saves the result to ```{your_file_name}_tc.txt```

### Arguments
1. path to the raw spectrum, expects two columns in the file
2. spectral resolution (optional, defaults to 10 angstroms)

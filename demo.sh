#!/bin/bash

# Copyright (c) 2016, Shane Frederic F. Carr
# 
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
# 
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# This is a black-box starter script.  This script is designed for MPI and
# GPAW, but it can be customized.  Read the README for more info.

if [[ -z "$1" ]]; then
	print "usage: ./demo.sh <number of processors>";
	exit 1;
fi

np = $1

# No MPI needed for step 2!
mpirun -np $np gpaw-python demo_01_run_bayes_opt.py
python demo_02_fit_length_scales.py
mpirun -np $np gpaw-python demo_03_obtain_training_data.py
mpirun -np $np gpaw-python demo_04_run_bayes_opt.py

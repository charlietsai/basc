#!/usr/bin/env python

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

"""
BASC is split into four steps:

  1. Generate an empty, relaxed surface
  2. Fit the length scales for the GP
  3. Obtain training data for accelerated DFT calculations (optional)
  4. Run the bayesian optimization

In general, step 1 is the slowest, because it requires performing a full
relaxation with DFT and BFGS (appx 40 calculations).  Step 2 is quick since
we can use a Lennard-Jones potential.  Step 3 is required to use the
"GPAWTrained" calculator for accelerated DFT calculations; it requires
several (at least 5) full DFT calculations.  Step 4 is the actual
optimization step; it is reasonably quick if using GPAWTrained, and somewhat
slower if using a normal DFT calculator (at least 50-100 calculations,
depending on the characteristics and dimensionality of the system).

You should use the procedure with four standalone steps if you intend to run
BASC multiple times on the same surface (e.g., to tune the BASC parameters or
run BASC with multiple different adsorbate molecules).  However, you can run
them in sequence in a single job if you want a quick start black box.
"""


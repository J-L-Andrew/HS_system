# Voro++ makefile
#
# Author : Chris H. Rycroft (LBL / UC Berkeley)
# Email  : chr@alum.mit.edu
# Date   : August 30th 2011

# Load the common configuration file
include ../config.mk

# List of executables
EXECUTABLES=import

# Makefile rules
all: $(EXECUTABLES)

import: import.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o import import.cc -lvoro++

clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean

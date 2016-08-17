warning_flags = -Wall -Wextra -Wno-unused-parameter -Wshadow
common_cflags = -O2 $(MFLAGS) $(STACKREALIGN) $(warning_flags)
AM_CFLAGS = -std=c99 $(common_cflags)

AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libfluxsmooth.la

libfluxsmooth_la_SOURCES = src/fluxsmooth.c

libfluxsmooth_la_LDFLAGS = -no-undefined -avoid-version $(PLUGINLDFLAGS)

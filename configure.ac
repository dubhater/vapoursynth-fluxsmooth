AC_INIT([fluxsmooth], [2], [https://github.com/dubhater/vapoursynth-fluxsmooth/issues], [fluxsmooth], [https://github.com/dubhater/vapoursynth-fluxsmooth/])

: ${CFLAGS=""}

AM_INIT_AUTOMAKE([foreign no-dist-gzip dist-xz subdir-objects no-define])
AM_SILENT_RULES([yes])

LT_INIT([disable-static win32-dll])

AC_CANONICAL_HOST

AC_PROG_CC

X86="false"

AS_CASE(
  [$host_cpu],
  [i?86],   [BITS="32" X86="true"],
  [x86_64], [BITS="64" X86="true"]
)

AS_CASE(
   [$host_os],
   [cygwin*|mingw*],
   [
      AS_IF(
         [test "x$BITS" = "x32"],
         [
          AC_SUBST([PLUGINLDFLAGS], ["-Wl,--kill-at"])
          AC_SUBST([STACKREALIGN], ["-mstackrealign"])
         ]
      )
   ]
)

AS_IF(
      [test "x$X86" = "xtrue"],
      [
       AC_DEFINE([FLUXSMOOTH_X86])

       AC_SUBST([MFLAGS], ["-mfpmath=sse -msse2"])
      ]
)

AM_CONDITIONAL([FLUXSMOOTH_X86], [test "x$X86" = "xtrue"])


PKG_CHECK_MODULES([VapourSynth], [vapoursynth])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

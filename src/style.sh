#!/bin/sh
cd `dirname $0` #lol
clang-format -style="{BasedOnStyle: Mozilla, UseTab: Never, IndentWidth: 4, TabWidth: 4, AccessModifierOffset: -4, PointerAlignment: Right}" -verbose -i *.cpp *.h *.hpp *.cu *.cuh

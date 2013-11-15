#!/bin/sh
# Create a 3GB Ramdisk for fast .dat file I/O

diskutil erasevolume HFS+ "RamDisk" `hdiutil attach -nomount ram://6291456`

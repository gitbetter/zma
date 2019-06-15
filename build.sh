#!/bin/bash

dev=false
xcode=false

while test $# -gt 0; do
	case "$1" in
		-d|--development) dev=true; shift;;
		-x|--xcode) xcode=true; shift;;
		*) break;;
	esac
done

if [ $"dev" = true ]; then
	flags="--development"
fi

if [ $"xcode" = true]; then
	flags=" $flags --xcode"
fi

if [ ! -d build/ ] || [ ! -f build/Makefile ]; then
	sh rebuild.sh $flags
elif [ "$xcode" = false ]; then
	cd build/; make; cd ..;
fi

if [ "$xcode" = true ]; then
	open build/
fi

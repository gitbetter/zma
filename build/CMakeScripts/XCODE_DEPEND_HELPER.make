# DO NOT EDIT
# This makefile makes sure all linkable targets are
# up-to-date with anything they link to
default:
	echo "Do not invoke directly"

# Rules to remove targets that are older than anything to which they
# link.  This forces Xcode to relink the targets from scratch.  It
# does not seem to check these dependencies itself.
PostBuild.zma.Debug:
/Users/adr7000/zma/build/Debug/zma:
	/bin/rm -f /Users/adr7000/zma/build/Debug/zma


PostBuild.zma.Release:
/Users/adr7000/zma/build/Release/zma:
	/bin/rm -f /Users/adr7000/zma/build/Release/zma


PostBuild.zma.MinSizeRel:
/Users/adr7000/zma/build/MinSizeRel/zma:
	/bin/rm -f /Users/adr7000/zma/build/MinSizeRel/zma


PostBuild.zma.RelWithDebInfo:
/Users/adr7000/zma/build/RelWithDebInfo/zma:
	/bin/rm -f /Users/adr7000/zma/build/RelWithDebInfo/zma




# For each target create a dummy ruleso the target does not have to exist

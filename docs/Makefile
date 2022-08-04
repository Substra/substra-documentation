# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W --keep-going -n
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
SUBSTRAEXAMPLEDIR 		= source/auto_examples
SUBSTRADOCDIR 		= source/documentation/references
SUBSTRAFLEXAMPLEDIR	= source/substrafl_doc/examples
SUBSTRAFLDOCDIR	= source/substrafl_doc/api

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean livehtml livetheme

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild -b html $(SOURCEDIR) $(BUILDDIR)/html

livetheme:
	sphinx-autobuild -b html -a $(SOURCEDIR) $(BUILDDIR)/html --watch source/static/

clean:
	rm -rf $(BUILDDIR)
	rm -rf $(SUBSTRADOCDIR)
	rm -rf $(SUBSTRAFLDOCDIR)
	rm -rf $(SUBSTRAEXAMPLEDIR)
	rm -rf $(SUBSTRAFLEXAMPLEDIR)
# Delete the local worker folders in connect-documentation
	find .. -type d -name local-worker -prune -exec rm -rf {} \;
# Delete the tmp folders in connect-documentation
	find .. -type d -name tmp -prune -exec rm -rf {} \;

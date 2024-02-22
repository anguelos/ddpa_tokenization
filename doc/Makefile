export PYTHONPATH := "./src:../src/:../../src"
# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
STATIC_DIR    = source/_static
SRC_DIR       = ../../src

# Target for generating UML class diagram
uml_diagram:
	@echo "Generating UML class diagram"
	@echo PWD: $(PWD)
	pyreverse -o png -p ddp_tokenization ddp_tkn -a 1 -f ALL  -d $(STATIC_DIR)


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: all clean doc shared help test

all: test shared

test:
	$(MAKE) -C tests test

doc:
	$(MAKE) -C doc all

shared:
	$(MAKE) -C src all

clean:
	$(MAKE) -C src clean
	$(MAKE) -C tests clean

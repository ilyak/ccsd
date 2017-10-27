CC= cc
CFLAGS= -g -Wall -Wextra -fopenmp -I$(LIBXM)
LDFLAGS= -L$(LIBXM) -L/usr/local/lib
LIBS= -lxm -lblas -lm

LIBXM= ../libxm/src

ccsd: ccsd.o
	$(CC) -o $@ $(CFLAGS) ccsd.o $(LDFLAGS) $(LIBS)

check: ccsd
	./ccsd -o 15 -v 31 -b 7

clean:
	rm -f ccsd ccsd.o ccsd.core xmpagefile

.PHONY: check clean

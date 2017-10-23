CC= cc
CFLAGS= -g -Wall -Wextra -fopenmp -I../libxm/src
LDFLAGS= -L../libxm/src -L/usr/local/lib
LIBS= -lxm -lmyblas -lgfortran

ccsd: ccsd.o
	$(CC) -o $@ $(CFLAGS) ccsd.o $(LDFLAGS) $(LIBS)

check: ccsd
	./ccsd -b 7 -o 15 -v 31

clean:
	rm -f ccsd ccsd.o ccsd.core xmpagefile

.PHONY: check clean

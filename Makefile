CC= cc
CFLAGS= -g -Wall -Wextra -fopenmp -I$(LIBXM)
LDFLAGS= -L$(LIBXM) -L/usr/local/lib
LIBS= -lxm -lblas -lm

# Intel Compiler (release build)
#CC= icc
#CFLAGS= -DNDEBUG -Wall -O3 -fopenmp -mkl=sequential -I$(LIBXM)
#LDFLAGS= -L$(LIBXM)
#LIBS= -lxm -lm

# Intel Compiler with MPI (release build)
#CC= mpicc
#CFLAGS= -DXM_USE_MPI -DNDEBUG -Wall -O3 -fopenmp -mkl=sequential -I$(LIBXM)
#LDFLAGS= -L$(LIBXM)
#LIBS= -lxm -lm

LIBXM= ../libxm/src

ccsd: ccsd.o
	$(CC) -o $@ $(CFLAGS) ccsd.o $(LDFLAGS) $(LIBS)

check: ccsd
	./ccsd -o 15 -v 31 -b 7

clean:
	rm -f ccsd ccsd.o ccsd.core xmpagefile

.PHONY: check clean

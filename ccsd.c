#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef XM_USE_MPI
#include <mpi.h>
#endif

#include "xm.h"

static void
print(const char *fmt, ...)
{
	va_list ap;
	int rank = 0;

#ifdef XM_USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	if (rank != 0)
		return;
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
	fflush(stdout);
}

static time_t
timer_start(const char *title)
{
	print("%s... ", title);
	return time(NULL);
}

static void
timer_stop(time_t timer)
{
	print("done in %d sec\n", (int)(time(NULL) - timer));
}

static void
init_oo(size_t o, size_t v, xm_tensor_t *oo)
{
	size_t i, j;

	(void)v;

	for (i = 0; i < o; i++) {
	for (j = i; j < o; j++) {
		/* aa */
		xm_tensor_set_canonical_block(oo, xm_dim_2(i, j));
		/* bb */
		xm_tensor_set_derivative_block(oo, xm_dim_2(i+o, j+o),
		    xm_dim_2(i, j), xm_dim_2(0, 1), 1);
	}}
	for (i = 0; i < o; i++) {
	for (j = 0; j < i; j++) {
		/* aa */
		xm_tensor_set_derivative_block(oo, xm_dim_2(i, j),
		    xm_dim_2(j, i), xm_dim_2(1, 0), -1);
		/* bb */
		xm_tensor_set_derivative_block(oo, xm_dim_2(i+o, j+o),
		    xm_dim_2(j, i), xm_dim_2(1, 0), -1);
	}}
}

static void
init_ov(size_t o, size_t v, xm_tensor_t *ov)
{
	size_t i, a;

	for (i = 0; i < o; i++) {
	for (a = 0; a < v; a++) {
		/* aa */
		xm_tensor_set_canonical_block(ov, xm_dim_2(i, a));
		/* bb */
		xm_tensor_set_derivative_block(ov, xm_dim_2(i+o, a+v),
		    xm_dim_2(i, a), xm_dim_2(0, 1), 1);
	}}
}

static void
init_oooo(size_t o, size_t v, xm_tensor_t *oooo)
{
	size_t i, j, k, l;

	(void)v;

	for (i = 0; i < o; i++) {
	for (j = i; j < o; j++) {
	for (k = j; k < o; k++) {
	for (l = k; l < o; l++) {
		/* aaaa */
		xm_tensor_set_canonical_block(oooo, xm_dim_4(i,j,k,l));
		/* bbbb */
		xm_tensor_set_derivative_block(oooo, xm_dim_4(i+o,j+o,k+o,l+o),
		    xm_dim_4(i,j,k,l), xm_dim_4(0,1,2,3), 1);
		/* abba */
		xm_tensor_set_canonical_block(oooo, xm_dim_4(i,j+o,k+o,l));
		/* baab */
		xm_tensor_set_derivative_block(oooo, xm_dim_4(i+o,j,k,l+o),
		    xm_dim_4(i,j+o,k+o,l), xm_dim_4(0,1,2,3), 1);
		/* abab */
		xm_tensor_set_canonical_block(oooo, xm_dim_4(i,j+o,k,l+o));
		/* baba */
		xm_tensor_set_derivative_block(oooo, xm_dim_4(i+o,j,k+o,l),
		    xm_dim_4(i,j+o,k,l+o), xm_dim_4(0,1,2,3), 1);
	}}}}
	for (i = 0; i < o; i++) {
	for (j = 0; j < i; j++) {
	for (k = 0; k < j; k++) {
	for (l = 0; l < k; l++) {
		/* aaaa */
		xm_tensor_set_derivative_block(oooo, xm_dim_4(i,j,k,l),
		    xm_dim_4(l,k,j,i), xm_dim_4(3,2,1,0), 1);
		/* bbbb */
		xm_tensor_set_derivative_block(oooo, xm_dim_4(i+o,j+o,k+o,l+o),
		    xm_dim_4(l,k,j,i), xm_dim_4(3,2,1,0), 1);
		/* abba */
		/* baab */
		/* abab */
		/* baba */
	}}}}
}

static void
init_ooov(size_t o, size_t v, xm_tensor_t *ooov)
{
	size_t i, j, k, a;

	for (i = 0; i < o; i++) {
	for (j = i; j < o; j++) {
	for (k = 0; k < o; k++) {
	for (a = 0; a < v; a++) {
		/* aaaa */
		xm_tensor_set_canonical_block(ooov, xm_dim_4(i,j,k,a));
		/* bbbb */
		xm_tensor_set_derivative_block(ooov, xm_dim_4(i+o,j+o,k+o,a+v),
		    xm_dim_4(i,j,k,a), xm_dim_4(0,1,2,3), 1);
		/* abba */
		xm_tensor_set_canonical_block(ooov, xm_dim_4(i,j+o,k+o,a));
		/* baab */
		xm_tensor_set_derivative_block(ooov, xm_dim_4(i+o,j,k,a+v),
		    xm_dim_4(i,j+o,k+o,a), xm_dim_4(0,1,2,3), 1);
		/* abab */
		xm_tensor_set_canonical_block(ooov, xm_dim_4(i,j+o,k,a+v));
		/* baba */
		xm_tensor_set_derivative_block(ooov, xm_dim_4(i+o,j,k+o,a),
		    xm_dim_4(i,j+o,k,a+v), xm_dim_4(0,1,2,3), 1);
	}}}}
	for (i = 0; i < o; i++) {
	for (j = 0; j < i; j++) {
	for (k = 0; k < o; k++) {
	for (a = 0; a < v; a++) {
		/* aaaa */
		xm_tensor_set_derivative_block(ooov, xm_dim_4(i,j,k,a),
		    xm_dim_4(j,i,k,a), xm_dim_4(1,0,2,3), -1);
		/* bbbb */
		xm_tensor_set_derivative_block(ooov, xm_dim_4(i+o,j+o,k+o,a+v),
		    xm_dim_4(j,i,k,a), xm_dim_4(1,0,2,3), -1);
		/* abba */
		xm_tensor_set_derivative_block(ooov, xm_dim_4(i,j+o,k+o,a),
		    xm_dim_4(j,i+o,k,a+v), xm_dim_4(1,0,2,3), -1);
		/* baab */
		xm_tensor_set_derivative_block(ooov, xm_dim_4(i+o,j,k,a+v),
		    xm_dim_4(j,i+o,k,a+v), xm_dim_4(1,0,2,3), -1);
		/* abab */
		xm_tensor_set_derivative_block(ooov, xm_dim_4(i,j+o,k,a+v),
		    xm_dim_4(j,i+o,k+o,a), xm_dim_4(1,0,2,3), -1);
		/* baba */
		xm_tensor_set_derivative_block(ooov, xm_dim_4(i+o,j,k+o,a),
		    xm_dim_4(j,i+o,k+o,a), xm_dim_4(1,0,2,3), -1);
	}}}}
}

static void
init_ovov(size_t o, size_t v, xm_tensor_t *ovov)
{
	size_t i, j, a, b;

	for (i = 0; i < o; i++) {
	for (j = i; j < o; j++) {
	for (a = 0; a < v; a++) {
	for (b = 0; b < v; b++) {
		/* aaaa */
		xm_tensor_set_canonical_block(ovov, xm_dim_4(i,a,j,b));
		/* bbbb */
		xm_tensor_set_derivative_block(ovov, xm_dim_4(i+o,a+v,j+o,b+v),
		    xm_dim_4(i,a,j,b), xm_dim_4(0,1,2,3), 1);
		/* abba */
		xm_tensor_set_canonical_block(ovov, xm_dim_4(i,a+v,j+o,b));
		/* baab */
		xm_tensor_set_derivative_block(ovov, xm_dim_4(i+o,a,j,b+v),
		    xm_dim_4(i,a+v,j+o,b), xm_dim_4(0,1,2,3), 1);
		/* abab */
		xm_tensor_set_canonical_block(ovov, xm_dim_4(i,a+v,j,b+v));
		/* baba */
		xm_tensor_set_derivative_block(ovov, xm_dim_4(i+o,a,j+o,b),
		    xm_dim_4(i,a+v,j,b+v), xm_dim_4(0,1,2,3), 1);
	}}}}
	for (i = 0; i < o; i++) {
	for (j = 0; j < i; j++) {
	for (a = 0; a < v; a++) {
	for (b = 0; b < v; b++) {
		/* aaaa */
		xm_tensor_set_derivative_block(ovov, xm_dim_4(i,a,j,b),
		    xm_dim_4(j,b,i,a), xm_dim_4(1,0,3,2), 1);
		/* bbbb */
		xm_tensor_set_derivative_block(ovov, xm_dim_4(i+o,a+v,j+o,b+v),
		    xm_dim_4(j,b,i,a), xm_dim_4(1,0,3,2), 1);
		/* abba */
		xm_tensor_set_derivative_block(ovov, xm_dim_4(i,a+v,j+o,b),
		    xm_dim_4(j,b+v,i,a+v), xm_dim_4(1,0,3,2), 1);
		/* baab */
		xm_tensor_set_derivative_block(ovov, xm_dim_4(i+o,a,j,b+v),
		    xm_dim_4(j,b+v,i+o,a), xm_dim_4(1,0,3,2), 1);
		/* abab */
		xm_tensor_set_derivative_block(ovov, xm_dim_4(i,a+v,j,b+v),
		    xm_dim_4(j,b+v,i,a+v), xm_dim_4(1,0,3,2), 1);
		/* baba */
		xm_tensor_set_derivative_block(ovov, xm_dim_4(i+o,a,j+o,b),
		    xm_dim_4(j,b+v,i,a+v), xm_dim_4(1,0,3,2), 1);
	}}}}
}

static void
init_oovv(size_t o, size_t v, xm_tensor_t *oovv)
{
	size_t i, j, a, b;

	for (i = 0; i < o; i++) {
	for (j = i; j < o; j++) {
	for (a = 0; a < v; a++) {
	for (b = a; b < v; b++) {
		/* aaaa */
		xm_tensor_set_canonical_block(oovv, xm_dim_4(i,j,a,b));
		/* bbbb */
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i+o,j+o,a+v,b+v),
		    xm_dim_4(i,j,a,b), xm_dim_4(0,1,2,3), 1);
		/* abba */
		xm_tensor_set_canonical_block(oovv, xm_dim_4(i,j+o,a+v,b));
		/* baab */
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i+o,j,a,b+v),
		    xm_dim_4(i,j+o,a+v,b), xm_dim_4(0,1,2,3), 1);
		/* abab */
		xm_tensor_set_canonical_block(oovv, xm_dim_4(i,j+o,a,b+v));
		/* baba */
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i+o,j,a+v,b),
		    xm_dim_4(i,j+o,a,b+v), xm_dim_4(0,1,2,3), 1);
	}}}}
	for (i = 0; i < o; i++) {
	for (j = 0; j < i; j++) {
	for (a = 0; a < v; a++) {
	for (b = 0; b < a; b++) {
		/* aaaa */
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i,j,a,b),
		    xm_dim_4(j,i,b,a), xm_dim_4(1,0,3,2), 1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(j,i,a,b),
		    xm_dim_4(j,i,b,a), xm_dim_4(0,1,3,2), -1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i,j,b,a),
		    xm_dim_4(j,i,b,a), xm_dim_4(1,0,2,3), -1);
		/* bbbb */
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i+o,j+o,a+v,b+v),
		    xm_dim_4(j,i,b,a), xm_dim_4(1,0,3,2), 1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(j+o,i+o,a+v,b+v),
		    xm_dim_4(j,i,b,a), xm_dim_4(0,1,3,2), -1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i+o,j+o,b+v,a+v),
		    xm_dim_4(j,i,b,a), xm_dim_4(1,0,2,3), -1);
		/* abba */
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i,j+o,a+v,b),
		    xm_dim_4(j,i+o,b+v,a), xm_dim_4(1,0,3,2), 1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(j,i+o,a+v,b),
		    xm_dim_4(j,i+o,b,a+v), xm_dim_4(0,1,3,2), -1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i,j+o,b+v,a),
		    xm_dim_4(j,i+o,b,a+v), xm_dim_4(1,0,2,3), -1);
		/* baab */
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i+o,j,a,b+v),
		    xm_dim_4(j,i+o,b+v,a), xm_dim_4(1,0,3,2), 1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(j+o,i,a,b+v),
		    xm_dim_4(j,i+o,b,a+v), xm_dim_4(0,1,3,2), -1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i+o,j,b,a+v),
		    xm_dim_4(j,i+o,b,a+v), xm_dim_4(1,0,2,3), -1);
		/* abab */
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i,j+o,a,b+v),
		    xm_dim_4(j,i+o,b,a+v), xm_dim_4(1,0,3,2), 1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(j,i+o,a,b+v),
		    xm_dim_4(j,i+o,b+v,a), xm_dim_4(0,1,3,2), -1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i,j+o,b,a+v),
		    xm_dim_4(j,i+o,b+v,a), xm_dim_4(1,0,2,3), -1);
		/* baba */
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i+o,j,a+v,b),
		    xm_dim_4(j,i+o,b,a+v), xm_dim_4(1,0,3,2), 1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(j+o,i,a+v,b),
		    xm_dim_4(j,i+o,b+v,a), xm_dim_4(0,1,3,2), -1);
		xm_tensor_set_derivative_block(oovv, xm_dim_4(i+o,j,b+v,a),
		    xm_dim_4(j,i+o,b+v,a), xm_dim_4(1,0,2,3), -1);
	}}}}
}

static void
init_ovvv(size_t o, size_t v, xm_tensor_t *ovvv)
{
	size_t i, a, b, c;

	for (i = 0; i < o; i++) {
	for (a = 0; a < v; a++) {
	for (b = 0; b < v; b++) {
	for (c = b; c < v; c++) {
		/* aaaa */
		xm_tensor_set_canonical_block(ovvv, xm_dim_4(i,a,b,c));
		/* bbbb */
		xm_tensor_set_derivative_block(ovvv, xm_dim_4(i+o,a+v,b+v,c+v),
		    xm_dim_4(i,a,b,c), xm_dim_4(0,1,2,3), 1);
		/* abba */
		xm_tensor_set_canonical_block(ovvv, xm_dim_4(i,a+v,b+v,c));
		/* baab */
		xm_tensor_set_derivative_block(ovvv, xm_dim_4(i+o,a,b,c+v),
		    xm_dim_4(i,a+v,b+v,c), xm_dim_4(0,1,2,3), 1);
		/* abab */
		xm_tensor_set_canonical_block(ovvv, xm_dim_4(i,a+v,b,c+v));
		/* baba */
		xm_tensor_set_derivative_block(ovvv, xm_dim_4(i+o,a,b+v,c),
		    xm_dim_4(i,a+v,b,c+v), xm_dim_4(0,1,2,3), 1);
	}}}}
	for (i = 0; i < o; i++) {
	for (a = 0; a < v; a++) {
	for (b = 0; b < v; b++) {
	for (c = 0; c < b; c++) {
		/* aaaa */
		xm_tensor_set_derivative_block(ovvv, xm_dim_4(i,a,b,c),
		    xm_dim_4(i,a,c,b), xm_dim_4(0,1,3,2), -1);
		/* bbbb */
		xm_tensor_set_derivative_block(ovvv, xm_dim_4(i+o,a+v,b+v,c+v),
		    xm_dim_4(i,a,c,b), xm_dim_4(0,1,3,2), -1);
		/* abba */
		xm_tensor_set_derivative_block(ovvv, xm_dim_4(i,a+v,b+v,c),
		    xm_dim_4(i,a+v,c,b+v), xm_dim_4(0,1,3,2), -1);
		/* baab */
		xm_tensor_set_derivative_block(ovvv, xm_dim_4(i+o,a,b,c+v),
		    xm_dim_4(i,a+v,c,b+v), xm_dim_4(0,1,3,2), -1);
		/* abab */
		xm_tensor_set_derivative_block(ovvv, xm_dim_4(i,a+v,b,c+v),
		    xm_dim_4(i,a+v,c+v,b), xm_dim_4(0,1,3,2), -1);
		/* baba */
		xm_tensor_set_derivative_block(ovvv, xm_dim_4(i+o,a,b+v,c),
		    xm_dim_4(i,a+v,c+v,b), xm_dim_4(0,1,3,2), -1);
	}}}}
}

int
main(int argc, char **argv)
{
	xm_allocator_t *allocator;
	xm_block_space_t *bsoo, *bsov, *bsvv;
	xm_block_space_t *bsoooo, *bsooov, *bsovov, *bsoovv, *bsovvv, *bsvvvv;
	xm_tensor_t *f_oo, *f_ov, *f_vv, *f1_vv, *f2_oo, *f2_ov, *f2_vv;
	xm_tensor_t *f3_oo, *t1, *t1new;
	xm_tensor_t *i_oooo, *i4_oooo, *i_ooov, *i2a_ooov, *i_ovov, *i1a_ovov;
	xm_tensor_t *i_oovv, *tt_oovv, *i_ovvv, *i_vvvv, *t2, *t2new;
	size_t ob, vb, o, v;
	int type = XM_SCALAR_DOUBLE;
	time_t timer;

#ifdef XM_USE_MPI
	MPI_Init(&argc, &argv);
#endif
	if (argc != 3) {
		print("usage: ccsd ob vb\n");
#ifdef XM_USE_MPI
		MPI_Finalize();
#endif
		return 1;
	}
	ob = (size_t)strtoll(argv[1], NULL, 10);
	vb = (size_t)strtoll(argv[2], NULL, 10);
	o = 32 * ob;
	v = 32 * vb;
	print("CCSD, C1, o = %zu, v = %zu\n", o, v);

	timer = timer_start("creating the objects");
	allocator = xm_allocator_create("xmpagefile");

	bsoo = xm_block_space_create(xm_dim_2(2*o, 2*o));
	bsov = xm_block_space_create(xm_dim_2(2*o, 2*v));
	bsvv = xm_block_space_create(xm_dim_2(2*v, 2*v));
	bsoooo = xm_block_space_create(xm_dim_4(2*o, 2*o, 2*o, 2*o));
	bsooov = xm_block_space_create(xm_dim_4(2*o, 2*o, 2*o, 2*v));
	bsovov = xm_block_space_create(xm_dim_4(2*o, 2*v, 2*o, 2*v));
	bsoovv = xm_block_space_create(xm_dim_4(2*o, 2*o, 2*v, 2*v));
	bsovvv = xm_block_space_create(xm_dim_4(2*o, 2*v, 2*v, 2*v));
	bsvvvv = xm_block_space_create(xm_dim_4(2*v, 2*v, 2*v, 2*v));

	xm_block_space_autosplit(bsoo);
	xm_block_space_autosplit(bsov);
	xm_block_space_autosplit(bsvv);
	xm_block_space_autosplit(bsoooo);
	xm_block_space_autosplit(bsooov);
	xm_block_space_autosplit(bsovov);
	xm_block_space_autosplit(bsoovv);
	xm_block_space_autosplit(bsovvv);
	xm_block_space_autosplit(bsvvvv);

	f_oo = xm_tensor_create(bsoo, type, allocator);
	f_ov = xm_tensor_create(bsov, type, allocator);
	f_vv = xm_tensor_create(bsvv, type, allocator);
	f1_vv = xm_tensor_create(bsvv, type, allocator);
	f2_oo = xm_tensor_create(bsoo, type, allocator);
	f2_ov = xm_tensor_create(bsov, type, allocator);
	f2_vv = xm_tensor_create(bsvv, type, allocator);
	f3_oo = xm_tensor_create(bsoo, type, allocator);
	t1 = xm_tensor_create(bsov, type, allocator);
	t1new = xm_tensor_create(bsov, type, allocator);
	i_oooo = xm_tensor_create(bsoooo, type, allocator);
	i4_oooo = xm_tensor_create(bsoooo, type, allocator);
	i_ooov = xm_tensor_create(bsooov, type, allocator);
	i2a_ooov = xm_tensor_create(bsooov, type, allocator);
	i_ovov = xm_tensor_create(bsovov, type, allocator);
	i1a_ovov = xm_tensor_create(bsovov, type, allocator);
	i_oovv = xm_tensor_create(bsoovv, type, allocator);
	tt_oovv = xm_tensor_create(bsoovv, type, allocator);
	i_ovvv = xm_tensor_create(bsovvv, type, allocator);
	i_vvvv = xm_tensor_create(bsvvvv, type, allocator);
	t2 = xm_tensor_create(bsoovv, type, allocator);
	t2new = xm_tensor_create(bsoovv, type, allocator);

	xm_block_space_free(bsoo);
	xm_block_space_free(bsov);
	xm_block_space_free(bsvv);
	xm_block_space_free(bsoooo);
	xm_block_space_free(bsooov);
	xm_block_space_free(bsovov);
	xm_block_space_free(bsoovv);
	xm_block_space_free(bsovvv);
	xm_block_space_free(bsvvvv);

	init_oo(ob, vb, f_oo);
	init_ov(ob, vb, f_ov);
	init_oo(vb, ob, f_vv);
	init_oo(vb, ob, f1_vv);
	init_oo(ob, vb, f2_oo);
	init_ov(ob, vb, f2_ov);
	init_oo(vb, ob, f2_vv);
	init_oo(ob, vb, f3_oo);
	init_ov(ob, vb, t1);
	init_ov(ob, vb, t1new);
	init_oooo(ob, vb, i_oooo);
	init_oooo(ob, vb, i4_oooo);
	init_ooov(ob, vb, i_ooov);
	init_ooov(ob, vb, i2a_ooov);
	init_ovov(ob, vb, i_ovov);
	init_ovov(ob, vb, i1a_ovov);
	init_oovv(ob, vb, i_oovv);
	init_oovv(ob, vb, tt_oovv);
	init_ovvv(ob, vb, i_ovvv);
	init_oooo(vb, ob, i_vvvv);
	init_oovv(ob, vb, t2);
	init_oovv(ob, vb, t2new);
	timer_stop(timer);

	timer = timer_start("filling the tensors");
	xm_set(f_oo, drand48());
	xm_set(f_ov, drand48());
	xm_set(f_vv, drand48());
	xm_set(f1_vv, drand48());
	xm_set(f2_oo, drand48());
	xm_set(f2_ov, drand48());
	xm_set(f2_vv, drand48());
	xm_set(f3_oo, drand48());
	xm_set(t1, drand48());
	xm_set(t1new, drand48());
	xm_set(i_oooo, drand48());
	xm_set(i4_oooo, drand48());
	xm_set(i_ooov, drand48());
	xm_set(i2a_ooov, drand48());
	xm_set(i_ovov, drand48());
	xm_set(i1a_ovov, drand48());
	xm_set(i_oovv, drand48());
	xm_set(tt_oovv, drand48());
	xm_set(i_ovvv, drand48());
	xm_set(i_vvvv, drand48());
	xm_set(t2, drand48());
	xm_set(t2new, drand48());
	timer_stop(timer);

	timer = timer_start("running one ccsd iteration");
	/* f1_vv */
	xm_copy(f1_vv, 1, f_vv, "ab", "ab");
	xm_contract(-0.5, i_oovv, t2, 1, f1_vv, "abcd", "abed", "ec");
	xm_contract(1, i_ovvv, t1, 1, f1_vv, "abcd", "ac", "bd");
	/* f2_ov */
	xm_copy(f2_ov, 1, f_ov, "ia", "ia");
	xm_contract(1, t1, i_oovv, 1, f2_ov, "ab", "cadb", "cd");
	/* f3_oo */
	xm_copy(f3_oo, 1, f_oo, "ij", "ij");
	xm_contract(1, f2_ov, t1, 1, f3_oo, "ab", "cb", "ac");
	xm_contract(0.5, i_oovv, t2, 1, f3_oo, "abcd", "ebcd", "ae");
	xm_contract(1, i_ooov, t1, 1, f3_oo, "abcd", "bd", "ac");
	/* t1 */
	xm_copy(t1new, 1, f_ov, "ia", "ia");
	xm_contract(1, f1_vv, t1, 1, t1new, "ab", "cb", "ca");
	xm_contract(-1, f3_oo, t1, 1, t1new, "ab", "ac", "bc");
	xm_contract(-1, i_ovov, t1, 1, t1new, "abcd", "cb", "ad");
	xm_contract(1, t2, f2_ov, 1, t1new, "abcd", "bd", "ac");
	xm_contract(0.5, i_ovvv, t2, 1, t1new, "abcd", "aecd", "eb");
	xm_contract(-0.5, i_ooov, t2, 1, t1new, "abcd", "abed", "ce");
	/* f2_oo */
	xm_contract(1, t1, t1, 0, i1a_ovov, "ab", "cd", "abcd");
	xm_copy(f2_oo, 1, f_oo, "ij", "ij");
	xm_contract(1, f_ov, t1, 1, f2_oo, "ab", "cb", "ca");
	xm_contract(1, i_ooov, t1, 1, f2_oo, "abcd", "bd", "ca");
	xm_contract(1, i_oovv, i1a_ovov, 1, f2_oo, "abcd", "ecbd", "ea");
	xm_contract(0.5, i_oovv, t2, 1, f2_oo, "abcd", "ebcd", "ea");
	/* f2_vv */
	xm_contract(1, t1, t1, 0, i1a_ovov, "ab", "cd", "abcd");
	xm_copy(f2_vv, 1, f1_vv, "ab", "ab");
	xm_contract(-1, f_ov, t1, 1, f2_vv, "ab", "ac", "cb");
	xm_contract(-1, i_oovv, i1a_ovov, 1, f2_vv, "abcd", "aebd", "ec");
	/* i1a_ovov */
	xm_copy(t2new, 1, t2, "ijab", "ijab");
	xm_contract(2, t1, t1, 1, t2new, "ab", "cd", "acbd");
	xm_copy(i1a_ovov, 1, i_ovov, "iajb", "iajb");
	xm_contract(-1, i_ovvv, t1, 1, i1a_ovov, "abcd", "ed", "abec");
	xm_contract(-1, i_ooov, t1, 1, i1a_ovov, "abcd", "be", "aecd");
	xm_contract(-0.5, t2new, i_oovv, 1, i1a_ovov, "abcd", "ebcf", "edaf");
	/* tt_oovv */
	xm_copy(tt_oovv, 1, t2, "ijab", "ijab");
	xm_contract(0.5, t1, t1, 1, tt_oovv, "ab", "cd", "acbd");
	/* i4_oooo */
	xm_copy(i4_oooo, 1, i_oooo, "abcd", "abcd");
	xm_contract(0.5, i_oovv, tt_oovv, 1, i4_oooo, "abcd", "efcd", "efab");
	xm_contract(1, i_ooov, t1, 1, i4_oooo, "abcd", "ed", "ceab");
	/* i2a_ooov */
	xm_copy(i2a_ooov, 1, i_ooov, "abcd", "abcd");
	xm_contract(-0.5, i4_oooo, t1, 1, i2a_ooov, "abcd", "de", "abce");
	xm_contract(0.5, tt_oovv, i_ovvv, 1, i2a_ooov, "abcd", "efcd", "abef");
	xm_contract(1, i_ovov, t1, 1, i2a_ooov, "abcd", "ed", "ceab");
	/* t2 */
	xm_copy(t2new, 1, i_oovv, "ijab", "ijab");
	xm_contract(1, t2, f2_vv, 1, t2new, "abcd", "ed", "abce");
	xm_contract(-1, i2a_ooov, t1, 1, t2new, "abcd", "ce", "abed");
	xm_contract(1, i1a_ovov, t2, 1, t2new, "abcd", "eafd", "cefb");
	xm_contract(1, i_ovvv, t1, 1, t2new, "abcd", "eb", "eadc");
	xm_contract(-1, t2, f2_oo, 1, t2new, "abcd", "eb", "aecd");
	xm_contract(0.5, i_vvvv, tt_oovv, 1, t2new, "abcd", "efcd", "efab");
	xm_contract(0.5, t2, i4_oooo, 1, t2new, "abcd", "efab", "efcd");
	/* energy */
	xm_contract(1, i_oovv, t1, 0, t1new, "abcd", "bd", "ac");
	timer_stop(timer);

	timer = timer_start("releasing the resources");
	xm_tensor_free_block_data(f_oo);
	xm_tensor_free_block_data(f_ov);
	xm_tensor_free_block_data(f_vv);
	xm_tensor_free_block_data(f1_vv);
	xm_tensor_free_block_data(f2_oo);
	xm_tensor_free_block_data(f2_ov);
	xm_tensor_free_block_data(f2_vv);
	xm_tensor_free_block_data(f3_oo);
	xm_tensor_free_block_data(t1);
	xm_tensor_free_block_data(t1new);
	xm_tensor_free_block_data(i_oooo);
	xm_tensor_free_block_data(i4_oooo);
	xm_tensor_free_block_data(i_ooov);
	xm_tensor_free_block_data(i2a_ooov);
	xm_tensor_free_block_data(i_ovov);
	xm_tensor_free_block_data(i1a_ovov);
	xm_tensor_free_block_data(i_oovv);
	xm_tensor_free_block_data(tt_oovv);
	xm_tensor_free_block_data(i_ovvv);
	xm_tensor_free_block_data(i_vvvv);
	xm_tensor_free_block_data(t2);
	xm_tensor_free_block_data(t2new);
	xm_tensor_free(f_oo);
	xm_tensor_free(f_ov);
	xm_tensor_free(f_vv);
	xm_tensor_free(f1_vv);
	xm_tensor_free(f2_oo);
	xm_tensor_free(f2_ov);
	xm_tensor_free(f2_vv);
	xm_tensor_free(f3_oo);
	xm_tensor_free(t1);
	xm_tensor_free(t1new);
	xm_tensor_free(i_oooo);
	xm_tensor_free(i4_oooo);
	xm_tensor_free(i_ooov);
	xm_tensor_free(i2a_ooov);
	xm_tensor_free(i_ovov);
	xm_tensor_free(i1a_ovov);
	xm_tensor_free(i_oovv);
	xm_tensor_free(tt_oovv);
	xm_tensor_free(i_ovvv);
	xm_tensor_free(i_vvvv);
	xm_tensor_free(t2);
	xm_tensor_free(t2new);
	xm_allocator_destroy(allocator);
	timer_stop(timer);
#ifdef XM_USE_MPI
	MPI_Finalize();
#endif
	return 0;
}

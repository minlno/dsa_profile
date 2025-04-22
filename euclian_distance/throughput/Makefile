all:
	gcc -O3 -std=gnu11 test.c -o test_exec \
  		-fopenmp -I. -laccel-config \
  		-I${MKLROOT}/include \
  		-L${MKLROOT}/lib/intel64 \
  		-lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
  		-lpthread -lm -ldl


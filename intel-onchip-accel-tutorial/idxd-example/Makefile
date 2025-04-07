CC=gcc
CFLAGS=-I. -I/home/mhkim/kernels/linux-ksm-batch/include/uapi
LDFLAGS = -laccel-config
DEPS = common.h
SRCS = offload.c single.c batch.c async.c
OBJS = $(SRCS:.c=.o)
EX = idxd-offload

.PHONY: clean

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(LDFLAGS)

$(EX): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)
	rm -f $(OBJS)

clean:
	rm -f $(EX) $(OBJS)

CC=g++
CFLAGS=-I.
LDFLAGS = -laccel-config -ldml
DEPS = common.h
SRCS = offload.cc single.cc batch.cc async.cc
OBJS = $(SRCS:.cc=.o)
EX = dml-offload

.PHONY: clean

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(LDFLAGS)

$(EX): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)
	rm -f $(OBJS)

clean:
	rm -f $(EX) $(OBJS)

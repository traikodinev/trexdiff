CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -Wpedantic -O2 -g -Iinclude -ftree-vectorize -ffast-math -fassociative-math -msse2

# CPU BLAS by default
LDFLAGS  = -lblas -lm

# enable NVBLASwith:
#   make USE_NVBLAS=1
ifeq ($(USE_NVBLAS),1)
	LDFLAGS  = -Wl,--no-as-needed -lnvblas -Wl,--as-needed -lcudart -lblas -lm
endif

BUILD    = build
LIB_SRCS = src/trexdiff.c src/tensor2d.c
LIB_HDRS = include/trexdiff.h include/tensor2d.h
LIB_OBJS = $(BUILD)/trexdiff.o $(BUILD)/tensor2d.o
LIB_SO   = $(BUILD)/libtrexdiff.so
EXAMPLE  = $(BUILD)/example

TEST_SRCS = tests/test_graph.c tests/test_graph_2d.c tests/test_tensor.c
TEST_BINS = $(TEST_SRCS:tests/%.c=$(BUILD)/%)

LEAK_BIN  = $(BUILD)/test_memory

# Path inside the Python package where the ctypes loader expects the .so
PY_SO   = python/trexdiff/libtrexdiff.so

.PHONY: all example python test test-leak clean

all: example python

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c $(LIB_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB_SO): $(LIB_SRCS) $(LIB_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC $(LIB_SRCS) -o $@ $(LDFLAGS)

example: $(EXAMPLE)

$(EXAMPLE): examples/example.c $(LIB_OBJS) $(LIB_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

python: $(PY_SO)

$(PY_SO): $(LIB_SO)
	cp $< $@

test: $(TEST_BINS) test-leak
	@for t in $(TEST_BINS); do echo "--- $$t ---"; $$t || exit 1; done

test-leak: $(LEAK_BIN)
	@echo "--- leak check: $(LEAK_BIN) ---"
	$(LEAK_BIN)

$(LEAK_BIN): tests/test_memory.c $(LIB_OBJS) $(LIB_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -fsanitize=leak -Itests $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(BUILD)/%: tests/%.c $(LIB_OBJS) $(LIB_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -Itests $< $(LIB_OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD)
	rm -f $(PY_SO)

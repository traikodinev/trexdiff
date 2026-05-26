CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -Wpedantic -O2 -g -Iinclude
LDFLAGS = -lm

BUILD    = build
LIB_SRC  = src/trexdiff.c
LIB_HDR  = include/trexdiff.h
LIB_OBJ  = $(BUILD)/trexdiff.o
LIB_SO   = $(BUILD)/libtrexdiff.so
EXAMPLE  = $(BUILD)/example

TEST_SRCS = tests/test_graph.c
TEST_BINS = $(TEST_SRCS:tests/%.c=$(BUILD)/%)

LEAK_BIN  = $(BUILD)/test_memory

# Path inside the Python package where the ctypes loader expects the .so
PY_SO   = python/trexdiff/libtrexdiff.so

.PHONY: all example python test test-leak clean

all: example python

$(BUILD):
	mkdir -p $(BUILD)

$(LIB_OBJ): $(LIB_SRC) $(LIB_HDR) | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB_SO): $(LIB_SRC) $(LIB_HDR) | $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC $< -o $@ $(LDFLAGS)

example: $(EXAMPLE)

$(EXAMPLE): examples/example.c $(LIB_OBJ) $(LIB_HDR) | $(BUILD)
	$(CC) $(CFLAGS) $< $(LIB_OBJ) -o $@ $(LDFLAGS)

python: $(PY_SO)

$(PY_SO): $(LIB_SO)
	cp $< $@

test: $(TEST_BINS) test-leak
	@for t in $(TEST_BINS); do echo "--- $$t ---"; $$t || exit 1; done

test-leak: $(LEAK_BIN)
	@echo "--- leak check: $(LEAK_BIN) ---"
	$(LEAK_BIN)

$(LEAK_BIN): tests/test_memory.c $(LIB_OBJ) $(LIB_HDR) | $(BUILD)
	$(CC) $(CFLAGS) -fsanitize=leak -Itests $< $(LIB_OBJ) -o $@ $(LDFLAGS)

$(BUILD)/%: tests/%.c $(LIB_OBJ) $(LIB_HDR) | $(BUILD)
	$(CC) $(CFLAGS) -Itests $< $(LIB_OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD)
	rm -f $(PY_SO)

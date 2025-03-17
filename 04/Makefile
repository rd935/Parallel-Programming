OPT    := -O2
DEBUG  := -g
COMP   := g++ $(OPT) $(DEBUG)

EXE    := 01_simple_eratosthenes 02_better_eratosthenes 03_bitwise_eratosthenes

all: $(EXE)

01_simple_eratosthenes: 01_simple_eratosthenes.cpp
	$(COMP) $< -o $@

01_better_eratosthenes: 02_better_eratosthenes.cpp
	$(COMP) $< -o $@

03_bitwise_eratosthenes: 03_bitwise_eratosthenes.cpp
	$(COMP) -mpopcnt $< -o $@  # Adding the -mpopcnt flag here

# Clean rule to remove the executables
clean:
	rm -f $(EXE)

# Benchmark rule with large input
n := 1000000000
bench:
	time ./01_simple_eratosthenes $(n)
	time ./02_better_eratosthenes $(n)
	time ./03_bitwise_eratosthenes $(n)

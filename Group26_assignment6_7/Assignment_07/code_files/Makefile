CXX = g++
CXXFLAGS = -O2 -fopenmp
LDFLAGS = -fopenmp

SRCS = main.cpp init.cpp utils.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = interpolation

INPUT_MAKER_SRC = input_file_maker.cpp
INPUT_MAKER = input_file_maker

.PHONY: all clean input_maker test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

input_maker: $(INPUT_MAKER_SRC)
	$(CXX) -O2 -o $(INPUT_MAKER) $<

# Test against provided test data
test: $(TARGET)
	./$(TARGET) ../Test_input.bin
	@echo "--- Comparing output with expected ---"
	diff -q Mesh.out ../Test_Mesh.out && echo "PASS: Output matches!" || echo "FAIL: Output differs!"

clean:
	rm -f $(OBJS) $(TARGET) $(INPUT_MAKER) Mesh.out input.bin

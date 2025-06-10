# Detect OS
UNAME_S := $(shell uname -s)

# Defaults
CXX := g++
CXXFLAGS := -std=c++17 -Iinclude -Wall -fopenmp -O3 -march=native -funroll-loops -ffast-math
LDFLAGS := -fopenmp

# macOS-specific paths (Homebrew LLVM + libomp)
ifeq ($(UNAME_S), Darwin)
    CXX := /opt/homebrew/opt/llvm/bin/clang++
    CXXFLAGS += -I/opt/homebrew/opt/libomp/include
    LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp
endif

# CPP files
SRC := $(shell find src -name '*.cpp')
OBJ := $(SRC:.cpp=.o)
TARGET := mlp

# Main
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(TARGET) $(OBJ)


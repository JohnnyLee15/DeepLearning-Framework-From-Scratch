# Compiler
CXX := /opt/homebrew/opt/llvm/bin/clang++

# Flags
CXXFLAGS := -std=c++11 -Iinclude -Wall -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp

# CPP files
SRC := $(wildcard src/**/*.cpp) Main.cpp

# Convert cpp files into .o files
OBJ := $(SRC:.cpp=.o)

# Target
TARGET := mlp

# Main
all: $(TARGET)

# Build Main
$(TARGET) : $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)


# Build object files
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean object files and mlp
clean:
	rm -f $(TARGET) $(OBJ)

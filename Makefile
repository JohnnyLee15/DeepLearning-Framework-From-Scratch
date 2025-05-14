# Compiler
CXX := clang++

# Flags
CXXFLAGS := -std=c++11 -Iinclude -Wall

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
	$(CXX) $(OBJ) -o $(TARGET)


# Build object files
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean object files and mlp
clean:
	rm -f $(TARGET) $(OBJ)

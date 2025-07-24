# Detect OS
UNAME_S := $(shell uname -s)

# Defaults
CXX := g++
CXXFLAGS := -std=c++17 \
			-isystem include/stb \
			-Iinclude \
			-Wall -fopenmp -O3 -march=native -funroll-loops -ffast-math
LDFLAGS := -fopenmp

# Source files: always .cpp
CPP_SRC := $(shell find src -name '*.cpp')
CPP_OBJ := $(CPP_SRC:.cpp=.o)

# macOS specifics
ifeq ($(UNAME_S), Darwin)
	SYSROOT  := $(shell xcrun --show-sdk-path)
	CXX      := /opt/homebrew/opt/llvm/bin/clang++
	CXXFLAGS += -isysroot $(SYSROOT) -I/opt/homebrew/opt/libomp/include
    LDFLAGS  := -isysroot $(SYSROOT) -L/opt/homebrew/opt/libomp/lib -lomp \
                -framework Metal -framework Foundation -framework MetalPerformanceShaders

	# Add Metal files
	METAL_SRC := $(shell find src/metal -name '*.metal')
	METAL_AIR := $(METAL_SRC:.metal=.air)
	METAL_LIB := CoreKernels.metallib

	# Add .mm files too
	METALFLAGS := -O3 -ffast-math -funroll-loops
	MM_SRC := $(shell find src -name '*.mm')
	MM_OBJ := $(MM_SRC:.mm=_gpu.o)

	OBJ := $(CPP_OBJ) $(MM_OBJ)
else
	OBJ := $(CPP_OBJ)
endif


TARGET := mlp

# Main
all: $(TARGET)

ifeq ($(UNAME_S), Darwin)
$(TARGET): $(OBJ) $(METAL_LIB)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)
else
$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)
endif

ifeq ($(UNAME_S), Darwin)
%_gpu.o: %.mm
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.air: %.metal
	xcrun -sdk macosx metal $(METALFLAGS) -c $< -o $@

$(METAL_LIB): $(METAL_AIR)
	xcrun -sdk macosx metallib $^ -o $@
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(TARGET) $(CPP_OBJ)
ifeq ($(UNAME_S), Darwin)
	rm -f $(MM_OBJ) $(METAL_AIR) $(METAL_LIB)
endif

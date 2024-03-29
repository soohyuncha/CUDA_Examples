# CUDA Compiler
NVCC = nvcc

# CUDA library
CC_FLAGS += -lcublas

# Source files
SOURCES = cublas_gemm.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# Target
TARGET = gemm

all: $(TARGET)

# $@ is target
$(TARGET): $(OBJECTS)
	$(NVCC) $(CC_FLAGS) -o $@ $(OBJECTS)
	rm -f $(OBJECTS)

%.o: %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f *.o
	rm -f $(TARGET)

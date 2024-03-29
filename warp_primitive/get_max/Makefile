# CUDA Compiler
NVCC = nvcc

# Source files
SOURCES = get_max.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# Target
TARGET = get_max

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS)
	rm -f $(OBJECTS)

%.o: %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f *.o
	rm -f $(TARGET)

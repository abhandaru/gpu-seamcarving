# macros
CC =g++
NVCC = nvcc
CFLAGS = -c -Wall -g -I/opt/local/include -I$(HOME)/cppunit/include
LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib
SOURCES = driver.cpp seamcarver.cu image.cpp energies.cu
OBJECTS = driver.o seamcarver.o image.o energies.o
LIBS = -lcppunit -ldl
EXECUTABLE = driver.out

# targets
all: $(SOURCES) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

  
%.o : %.cu %.h
	$(NVCC) -c $< -o $@
  
  
.cpp.o:
	$(CC) $(CFLAGS) $< -o $@


# clean
clean:
	rm -f $(EXECUTABLE) $(OBJECTS)

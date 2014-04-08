# macros
CC=g++
CFLAGS=-c -Wall -g
LDFLAGS=
SOURCES=driver.cpp seamcarver.cpp image.cpp energies.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=driver.out

# targets
all: $(SOURCES) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@


.cpp.o:
	$(CC) $(CFLAGS) $< -o $@


# clean
clean:
	rm -f $(EXECUTABLE) $(OBJECTS)

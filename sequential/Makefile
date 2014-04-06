# macros
CC=g++
CFLAGS=-c -Wall
LDFLAGS=
SOURCES=driver.cpp seamcarver.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=driver

# targets
all: $(SOURCES) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@


.cpp.o:
	$(CC) $(CFLAGS) $< -o $@


# clean
clean:
	rm -f $(EXECUTABLE) $(OBJECTS)

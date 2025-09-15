# Default compiler
CC ?= gcc
STRIP ?= strip

# Compiler flags
CFLAGS = -Wall -Wextra -std=c99 -O3 -flto
#CFLAGS += -DLOREM_IPSUM_DEBUG=1
DEPFLAGS = -MMD -MP

# Linker flags
LDFLAGS = -flto -O3

# Include directories
INCLUDES = -I. -I../c_lib/include

# Build directory
BUILDDIR = ../../data/build_cli

# Distribution directory
DISTDIR = ../../dist/cli

# Source directories
SRCDIR = .
LIBDIR = ../c_lib/src

# Detect Windows (MinGW/MSYS2)
ifeq ($(OS),Windows_NT)
    EXE_SUFFIX = .exe
else
    EXE_SUFFIX =
endif

# Find all .c files in current directory and models directory
SOURCES = $(wildcard $(SRCDIR)/*.c) $(wildcard $(LIBDIR)/*.c)

# Generate object file names (place in build directory)
OBJECTS = $(addprefix $(BUILDDIR)/, $(notdir $(SOURCES:.c=.o)))

# Generate dependency file names
DEPS = $(OBJECTS:.o=.d)

# Target executable name (based on the main source file)
TARGET = $(DISTDIR)/lorem-ipsum$(EXE_SUFFIX)

# Default target
all: $(BUILDDIR) $(DISTDIR) $(TARGET)

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Create distribution directory
$(DISTDIR):
	mkdir -p $(DISTDIR)

# Link the executable
$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
	$(STRIP) $@

# Compile .c files from current directory
$(BUILDDIR)/%.o: %.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

# Compile .c files from lib directory
$(BUILDDIR)/%.o: $(LIBDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

# Include dependency files (if they exist)
-include $(DEPS)

# Clean target
clean:
	rm -f $(OBJECTS) $(DEPS) $(TARGET)

# Clean everything including backup files and build directory
distclean: clean
	rm -rf $(BUILDDIR) $(DISTDIR)

# Phony targets
.PHONY: all clean distclean debug

# Print variables for debugging
debug:
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "EXE_SUFFIX: $(EXE_SUFFIX)"
	@echo "BUILDDIR: $(BUILDDIR)"
	@echo "DISTDIR: $(DISTDIR)"
	@echo "SOURCES: $(SOURCES)"
	@echo "OBJECTS: $(OBJECTS)"
	@echo "DEPS: $(DEPS)"
	@echo "TARGET: $(TARGET)"



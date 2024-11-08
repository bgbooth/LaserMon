
# Choice of compiler
CC=gcc

# CFLAGS will be the options passed to the compiler.
CFLAGS= -c -g -Wall -O2 -I/usr/include/hdf5/serial

all: demo

demo: main.o model.o hdf5_read.o model_params.o 
	$(CC) model_params.o model.o hdf5_read.o main.o -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lm -lhdf5 -o demo

model.o: model.c model.h
	$(CC) $(CFLAGS) model.c

model_params.o: model_params.c
	$(CC) $(CFLAGS) model_params.c

hdf5_read.o: hdf5_read.c hdf5_read.h
	$(CC) $(CFLAGS) hdf5_read.c

main.o: main.c 
	$(CC) $(CFLAGS) main.c

clean:
	rm -rf *.o *~ demo

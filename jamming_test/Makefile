XX=g++
SRCS=code/algebra.cpp\
		code/analysis.cpp\
     code/Node.cpp\
     code/Superball.cpp\
		 code/Voxel.cpp\
		 code/Cell.cpp

OBJS=$(SRCS:.cpp=.o)

EXEC=analysis

start:$(OBJS)
	$(XX) -O3 -o $(EXEC) $(OBJS)

.cpp.o:
	$(XX) -O3 -o $@ -c $< 

clean:
	rm $(OBJS)

XX=g++
SRCS=analysis.cpp\

OBJS=$(SRCS:.cpp=.o)

EXEC=analysis

start:$(OBJS)
	$(XX) -O3 -o $(EXEC) $(OBJS)

.cpp.o:
	$(XX) -O3 -o $@ -c $< 

clean:
	rm $(OBJS)
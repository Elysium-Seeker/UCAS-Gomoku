CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -I./src
SRC = src/main.c src/board.c src/game.c src/rules.c src/ai_engine.c
OBJ = $(SRC:.c=.o)
TARGET = gomoku

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(TARGET).exe $(TARGET)

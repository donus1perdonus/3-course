#ifndef TASK7_H
#define TASK7_H

#define MAX_PATH_LENGTH 1024
#define MAX_LINE_LENGTH 4096

int count_occurrences(const char* filename, const char* search_str);

void create_fork_bomb(int depth, int current_depth);

int task7_main(int argc, char* argv[]);

#endif // TASK7_H

/**
To compile: gcc -O0 -o a.out simple_example.c
Note: IACA markers must surround a SINGLE basic block (no branches/jumps)
 */
#include <stdio.h>
#include "iacaMarks.h"

int kernel() {
  volatile int acc = 1;  // Use volatile to prevent optimization
  IACA_START
    // Simple arithmetic operations - forms a single basic block
    // No loops, no branches, no function calls
    acc += 1;
    acc += 2;
    acc += 3;
  IACA_END
  return acc;
}

int main(int argc, char **argv) {
  printf("kernel: %d\n", kernel());
  return 0;
}
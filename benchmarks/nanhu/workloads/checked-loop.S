/* Bare-metal RV64I, no compressed instructions, self-checking RAM loop.
 * Success: x7 == LOOP_COUNT, x10 == 0, and XiangShan's custom trap retires.
 */
#ifndef LOOP_COUNT
#define LOOP_COUNT 1000
#endif
.section .text
.global _start
.option norvc
.option norelax
_start:
    addiw t1, zero, 1
    slli t1, t1, 31
    addi t1, t1, 512
    li t0, LOOP_COUNT
    addi t2, zero, 0
loop:
    addi t2, t2, 1
    xor t5, t2, t0
    sd t5, 0(t1)
    ld t4, 0(t1)
    xor t6, t4, t5
    bne t6, zero, failed
    addi t0, t0, -1
    bne t0, zero, loop
    fence rw, rw
    addi a0, zero, 0
    .word 0x0005006b
halt:
    jal zero, halt
failed:
    addi a0, zero, 1
    .word 0x0005006b
    jal zero, failed

# lower bounding the runtime

## N = 3

we have round 0, 1: cached
round 2, 3, 4, 5, 6, 7, 8, 9: load the next round
round 10: no load needed; wraparound
round 11, 12: cached
round 13, 14: load the next round
round 15: next loading not needed, since store

### load

- constant defs
  - 5 + 12 const loads
  - 3 vloads
- batch-wise:
  - mid rounds, 10 of these
    - 8 scalar loads
  - top of batch
    - 1 vload

this is 17 scalar + 3 vload for the entire kernel
and 10 * 8 scalar + 1 vload for a single batch
so 20 + 32\*81 load slots = 2612 slots, **1306 cycles**

if we cache one more round, theoretical optimum is **1050 cycles**

### alu

throughputs:

- regular vops: 7.5 vops/cycle
- vbroadcast, fma: 6 vops/cycle

- constant defs
  - 6 * 2 + 1 + 4 = 17 vbroadcasts
- batch-wise:
  - hash, 16 of these
    - each is 3 fmas, 9 vops
  - round 0/11: 2 vops
  - round 1/12: 3 vops
  - mid rounds (10 of these)
    - 4 vops, 1 fma
  - round 10, wraparound: 1
  - round 15, last: 1
- kernel-wide:
  - 2 + 2 \* 2 + 4 \* 2 = 14 vbroadcasts
  - 16 more for N = 4
- so 17 vbroadcast + 32\*(16\*3 + 10\*1) fmas + 14 vbroadcast = 1887 valu-specific ops = 314.5 cycles
- 32\*(16\*9 + 2\*2 + 2\*3 + 10\*4 + 1 + 1) = 6272 vops = 836.266 cycles
- in total, **1150.766 cycles**

### flow

- const defs:
  - 1 add_imm
- batch-wise:
  - initial load: 1 add_imm
  - round 0/11: 1 vselect
  - round 1/12: 4 vselect
  - round 15, last: 1 add_imm

1 + 32\*(1+2\*1+2\*4+1) = **385 cycles**

### ratio of valu to non-valu

- there are 1887 valu-specific ops, and 6272 vops
- if we want to achieve ratio of VALU:ALU of 6:12 = 1/2, we need:
- (1887 + 6272 - n)/8n = 1/2; n <= 6272; n is the number of vops we scalarize
- 2\*(8159-n) = 8n; n = 1631.8; /32 = 51 ops per batch; roughly 3 ops per hash (48)
  - can make it a little higher to get better util?
